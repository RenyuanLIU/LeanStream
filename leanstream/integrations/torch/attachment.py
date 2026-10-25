"""Manual module replacement: model owns computation around each sparse MLP."""
import torch
from functools import wraps
from .contracts import ModelPoints,ranked_ids
from .inspection import inspect_model,resolve
from .compute import Group2Bundle

# Preserve owners on a CUDA error for which the bridge cannot establish retirement.
_QUARANTINED=[]

class SparseMLP(torch.nn.Module):
    def __init__(self,original,owner,index,bundle):
        super().__init__();self.original=original;self.owner=owner;self.index=index;self.bundle=bundle;self.pending=False;self.future=None;self.trace=[]
    def decode(self,x):return x.ndim==3 and tuple(x.shape[:2])==(1,1)
    def predict(self,stage,x,source):
        value=stage.predictor(source,self.index,x,stage.budget)
        return value
    def publish(self,value,stage,source):
        if hasattr(value,'done'):
            if not value.done():return False
            value=value.result() # already completed; never wait for a newer prediction
        ids=ranked_ids(value,self.bundle.weights.group_count,stage.budget)
        self.bundle.native.update(ids,stage.io_budget)
        if self.bundle.compute.check_execution:self.trace.append(dict(event='update',source=source,ids=ids,**self.bundle.native.stats()))
        return ids
    def begin(self,x):
        if torch.is_grad_enabled():raise RuntimeError('Use torch.inference_mode(); training/autograd unsupported')
        if not self.decode(x):
            if x.ndim!=3 or x.shape[0]!=1:raise ValueError('Only batch1 [1,sequence,H] supported')
            return
        if x.device.type!='cuda' or x.dtype!=torch.float16:raise ValueError('Decode requires CUDA FP16')
        self.bundle.native.begin();self.pending=True;self.trace=[]
        early=self.owner.prediction.early
        if early:
            try:
                self.future=self.predict(early,x,'layer_input');self.publish(self.future,early,'layer_input')
            except BaseException:self.abort();raise
    def abort(self):
        if self.pending:self.bundle.native.abort();self.pending=False
    def forward(self,x):
        if not self.decode(x):return self.original(x)
        if not self.pending:raise RuntimeError('MLP must run inside its configured decoder layer')
        try:
            if x.dtype!=torch.float16 or x.device.type!='cuda' or x.shape[-1]!=self.bundle.weights.width:raise ValueError('MLP input geometry/dtype')
            final=self.owner.prediction.final
            if self.owner.mode=='progressive':
                if hasattr(self.future,'done') and self.future.done():self.publish(self.future,self.owner.prediction.early,'layer_input_refinement')
                # Submission precedes final prediction/finalization. GPU B waits on the public queue.
                self.bundle.enqueue(x);self.trace.append(dict(event='pre_enqueue'))
            value=self.predict(final,x,'mlp_input')
            ids=self.publish(value,final,'mlp_input')
            if ids is False:raise ValueError('Final prediction must already be CPU-ready; no synchronous future wait is introduced')
            if not set(self.bundle.policy.core)<=set(ids):raise ValueError('Final selection must include every configured core group')
            self.bundle.native.finalize(ids);self.trace.append(dict(event='finalize',ids=ids))
            if self.owner.mode=='basic':
                self.bundle.native.check(self.bundle.native.lib.lt_wait_io(self.bundle.native.context,5000))
                self.bundle.enqueue(x);self.trace.append(dict(event='basic_enqueue_after_io'))
            result=self.bundle.output(x.shape);self.bundle.native.commit();self.pending=False;return result
        except BaseException:self.abort();raise

class Attachment:
    def __init__(self,model,points,weights,compute,cache_policy,prediction,mode):
        if getattr(model,'_leanstream_attachment',None) is not None:raise ValueError('Model already attached')
        if model.training:raise ValueError('Call model.eval(); training is unsupported')
        report=inspect_model(model,points)
        if not report['compatible']:raise ValueError(f'Manual inspection required: {report}')
        points=report['points'];layers=resolve(model,points.layers)
        if not (len(weights)==len(cache_policy)==len(layers)):raise ValueError('One weight/policy specification per layer required')
        self.model=model;self.prediction=prediction;self.mode=mode;self.entries=[];self.hooks=[];self.guards=[];self.closed=False
        # Validate every layer before mutating any module.
        for row,w,p in zip(report['layers'],weights,cache_policy):
            w.validate();p.validate(w.group_count);prediction.validate(w.group_count,mode)
            if row['hidden']!=w.width or row['intermediate']!=w.group_count*2:raise ValueError('Packed weight/model dimensions differ')
        try:
            for i,(layer,w,p) in enumerate(zip(layers,weights,cache_policy)):
                original=resolve(layer,points.mlp);bundle=Group2Bundle(w,p,compute)
                replacement=SparseMLP(original,self,i,bundle)
                parent=resolve(layer,points.mlp.rsplit('.',1)[0]) if '.' in points.mlp else layer
                name=points.mlp.split('.')[-1];setattr(parent,name,replacement);self.entries.append((parent,name,original,replacement))
                def begin(module,args,kwargs,r=replacement):
                    x=args[0] if args else kwargs.get('hidden_states')
                    if not isinstance(x,torch.Tensor):raise ValueError('Decoder must expose hidden_states as first positional/keyword Tensor')
                    r.begin(x)
                original_forward=layer.forward
                @wraps(original_forward)
                def guarded(*args,_forward=original_forward,_r=replacement,**kwargs):
                    try:return _forward(*args,**kwargs)
                    finally:
                        if _r.pending:_r.abort()
                # Vendor Torch2.1 lacks always_call forward hooks. Delegate the
                # unchanged layer body through a cleanup guard instead.
                self.guards.append((layer,original_forward));layer.forward=guarded
                self.hooks.append(layer.register_forward_pre_hook(begin,with_kwargs=True))
            model._leanstream_attachment=self
        except BaseException:self.close();raise
    def refine(self,layer,ids,io_budget):
        r=self.entries[layer][3]
        if not r.pending:raise ValueError('No active layer to refine')
        ids=ranked_ids(ids,r.bundle.weights.group_count,r.bundle.weights.group_count)
        if not 0<=io_budget<=r.bundle.weights.group_count:raise ValueError('Invalid I/O budget')
        r.bundle.native.update(ids,io_budget);r.trace.append(dict(event='explicit_refine',ids=ids,**r.bundle.native.stats()))
    def drain(self):
        for *_,r in self.entries:r.bundle.native.drain()
    def reset(self):
        # No KV-cache ownership: application resets its own model cache.
        for *_,r in self.entries:r.abort()
        self.drain()
    def close(self):
        if self.closed:return
        for hook in self.hooks:hook.remove()
        self.hooks.clear()
        for layer,original_forward in self.guards:layer.forward=original_forward
        self.guards.clear()
        errors=[]
        for parent,name,original,r in reversed(self.entries):
            try:r.abort();r.bundle.close()
            except BaseException:
                _QUARANTINED.append(r.bundle);errors.append(r.bundle) # Keep all owners alive after unproved retirement.
            finally:setattr(parent,name,original)
        self.entries.clear();self.closed=True
        if getattr(self.model,'_leanstream_attachment',None) is self:del self.model._leanstream_attachment
        if errors:raise RuntimeError('GPU retirement failed; native/tensor owners retained in quarantine')
    detach=close
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

def attach_leanstream(model,*,points,weights,compute,cache_policy,prediction,mode='progressive'):
    """Return a lifecycle handle; continue calling the original model.forward/generate."""
    return Attachment(model,points,weights,compute,cache_policy,prediction,mode)
