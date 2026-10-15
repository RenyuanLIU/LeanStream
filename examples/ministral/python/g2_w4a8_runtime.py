"""Fixed G2 SSD decode; HF SDPA/RoPE/KV cache and dense GPTQ prefill remain HF-owned."""
import ctypes as C
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from python.predictor_cpu_native import Kernel, aligned
from python.v1_storage import PackedReader, gpu_dequantize
from scripts.predictor_shared_input import Bridge


def ms():return perf_counter()*1000


class GroupCache:
    """One slot per atomic group; preserve hits, replace only nonselected slots."""
    def __init__(self,validate=False):
        self.validate=validate
        self.ids=np.empty(1434,np.int32)
        self.group_to_slot=np.empty(7168,np.int32)
        self.wanted_mask=np.empty(7168,np.bool_)
        self.selected=np.empty(1434,np.int32)
        self.reset()
    def reset(self):
        self.ids.fill(-1);self.group_to_slot.fill(-1);self.wanted_mask.fill(False)
        self.keep=np.empty(0,np.int32)
    def plan(self,selected):
        selected=np.asarray(selected,np.int32)
        if selected.shape!=(1434,) or np.any(selected<0) or np.any(selected>=7168):
            raise ValueError('Invalid atomic selection')
        if self.validate and len(np.unique(selected))!=1434:
            raise ValueError('Invalid atomic selection')
        missing=selected[self.group_to_slot[selected]<0]
        self.wanted_mask.fill(False)
        self.wanted_mask[selected]=True
        # -1 indexes a valid mask element, but empty slots are always evictable.
        slots=np.flatnonzero((self.ids<0) | ~self.wanted_mask[self.ids])
        assert len(slots)==len(missing)
        old=self.ids[slots]
        self.group_to_slot[old[old>=0]]=-1
        self.ids[slots]=missing
        self.group_to_slot[missing]=slots
        if self.validate:
            assert np.array_equal(np.sort(self.ids),np.sort(selected))
            inverse=np.full(7168,-1,np.int32)
            inverse[self.ids]=np.arange(1434,dtype=np.int32)
            assert np.array_equal(self.group_to_slot,inverse)
        return missing,slots
    def commit_importance(self,importance):
        # Stable ordering is selected order, matching AtomicGroupState, not physical slot order.
        by_id={int(g):i for i,g in enumerate(self.ids)}
        values=np.asarray([importance[by_id[int(g)]] for g in self.selected])
        self.keep=np.ascontiguousarray(self.selected[np.argsort(-values,kind='stable')[:1219]],dtype=np.int32)


class Runtime:
    def __init__(self,model,root,predictor_library,shared_library=None,timing=False,validate=False):
        self.model=model;self.root=Path(root);self.timing=timing;self.validate=validate
        self.records=[];self.layers=[];self.bridge=None;self.kernel=None
        self.shared=shared_library is not None
        if self.shared:
            self.bridge=Bridge(shared_library,predictor_library)
            lib=self.bridge.lib;lib.si_dlpack.argtypes=[C.c_void_p];lib.si_dlpack.restype=C.c_void_p
            capsule=C.pythonapi.PyCapsule_New;capsule.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];capsule.restype=C.py_object
            self.mapped=torch.utils.dlpack.from_dlpack(capsule(lib.si_dlpack(self.bridge.context),b'dltensor',None))
            lib.si_profile_events.argtypes=[C.c_void_p]*3
        else:
            # Discrete RTX cannot use Tegra system atomics. Identical native integer arithmetic;
            # host copy/normal HF norm is an accuracy-only hardware adaptation, never a timing result.
            self.kernel=Kernel(predictor_library,4,12)
            self.quant=aligned(np.zeros(1664,np.int8))
        self.staging=torch.empty((2868,3,2048),dtype=torch.uint8,pin_memory=True)
        self.device_packed=torch.empty_like(self.staging,device='cuda')
        for i,layer in enumerate(model.model.layers):
            if type(layer.self_attn).__name__!='MistralSdpaAttention':raise ValueError('HF SDPA is required')
            wrapper=G2MLP(self,layer.mlp,i);self.layers.append(wrapper)
            layer.mlp=wrapper;layer.post_attention_layernorm=SharedNorm(layer.post_attention_layernorm,wrapper)
            layer.register_forward_pre_hook(wrapper.layer_begin)
            layer.register_forward_hook(wrapper.layer_end)
            layer.self_attn.register_forward_pre_hook(wrapper.attention_begin,with_kwargs=True)
            layer.self_attn.register_forward_hook(wrapper.attention_end,with_kwargs=True)
        self.reset()
    def reset(self):
        for layer in self.layers:layer.reset()
    @contextmanager
    def event(self,row,key):
        if not self.timing:yield;return
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();yield;b.record();row['_events'][key]=(a,b)
    def resolve(self):
        # Caller has completed the sample, outside predictor handoff and module critical regions.
        for row in self.records:
            for key,(a,b) in row.pop('_events',{}).items():row[key]=row.get(key,0)+a.elapsed_time(b)
            measured=sum(row.get(k,0) for k in ('attention','post_attention_norm','predictor_total',
                'selection_cache_bookkeeping','ssd_read','packed_h2d','gpu_dequant_cache_update','sparse_mlp','importance_keep_update'))
            row['other_layer']=row['layer_total']-measured
        return self.records


class SharedNorm(nn.Module):
    def __init__(self,original,mlp):
        super().__init__();self.original=original
        # Avoid module ownership cycles (MLP -> Runtime -> model).
        object.__setattr__(self,'mlp',mlp)
    def forward(self,x):
        m=self.mlp;r=m.runtime
        if m.prefill:return self.original(x)
        if x.shape!=(1,1,4096):raise ValueError('Decode must have q_len=1')
        row=m.row
        if r.shared:
            bridge=r.bridge
            if r.timing:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                # Initialize event handles before handing them to the native stream recorder.
                a.record();b.record()
                bridge.lib.si_profile_events(bridge.context,C.c_void_p(a.cuda_event),C.c_void_p(b.cuda_event))
                row['_events']['post_attention_norm']=(a,b)
            epoch=bridge.lib.si_epoch(bridge.context)
            item=dict(residual_gpu=x,norm=self.original,eps=self.original.variance_epsilon,
                      features=m.features,w=m.weights,scales=m.scales,keep=m.cache.keep)
            times=bridge.call('b',item,torch.cuda.current_stream().cuda_stream)
            assert bridge.lib.si_epoch(bridge.context)==epoch+1
            row.update(predictor_input_quant=float(times[1]),predictor_w4a8=float(times[2]),predictor_total=float(times[3]))
            m.fill=bridge.output.copy();value=r.mapped
            if r.validate and not m.input_checked:
                expected=self.original(x).detach().cpu().numpy().reshape(-1)
                np.testing.assert_allclose(bridge.host,expected,rtol=1e-3,atol=1e-3)
                m.input_checked=True
        else:
            value=self.original(x)
            full=value.detach().cpu().numpy().reshape(-1)
            selected=full[m.features].astype(np.float32)
            scale=np.float32(np.max(np.abs(selected))/np.float32(127))
            if scale==0:scale=np.float32(1)
            r.quant[:1638]=np.clip(np.rint(selected/scale),-127,127).astype(np.int8)
            m.fill=r.kernel.predict(m.weights,m.scales,r.quant,scale,m.cache.keep,skip=False).copy()
        return value


class G2MLP(nn.Module):
    def __init__(self,runtime,original,layer):
        super().__init__();self.original=original;object.__setattr__(self,'runtime',runtime)
        self.layer=layer;root=runtime.root;folder=root/'predictor'/f'layer_{layer:02d}'
        with np.load(folder/'mapping.npz') as z:
            self.groups=z['groups'];self.features=aligned(z['features'].astype(np.int32))
        self.weights=aligned(np.fromfile(folder/'w4.bin',np.uint8));self.scales=aligned(np.fromfile(folder/'scales4.bin',np.float32))
        assert self.weights.size==7168*2*832 and self.scales.size==7168*2
        self.reader=PackedReader(root/'g2_weights'/f'layer_{layer:02d}',0,2)
        assert np.array_equal(self.reader.mapping[self.groups.ravel()],np.arange(14336))
        self.reader.scales_on(torch.device('cuda'))
        self.rows=torch.empty((2868,3,4096),device='cuda',dtype=torch.float16)
        self.importance_host=torch.empty(1434,dtype=torch.float32,pin_memory=True)
        self.cache=GroupCache(validate=runtime.validate);self.input_checked=False;self.kv_checks=0
        original.down_proj.register_forward_pre_hook(self.capture)
    def reset(self):
        self.prefill=True;self.cache.reset();self.seed=None;self.previous_kv=None;self.calls=0
    def capture(self,module,inputs):
        if self.prefill:self.seed=inputs[0][0,-1].detach().abs().float().cpu().numpy()
    def layer_begin(self,module,inputs):
        self.was_prefill=self.prefill
        if not self.prefill:self.start=ms();self.row=dict(layer=self.layer,token=self.calls,_events={})
    def layer_end(self,module,inputs,output):
        if not self.was_prefill and self.runtime.timing:
            # One coarse layer completion fence, never inside Mode-B handoff or each GPU module.
            end=torch.cuda.Event();end.record();end.synchronize()
            self.row['layer_total']=ms()-self.start;self.runtime.records.append(self.row)
    def attention_begin(self,module,args,kwargs):
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        cache=kwargs.get('past_key_value')
        before=cache.get_seq_length(self.layer) if cache is not None else 0
        self.attention_before=int(before);self.attention_q=int(hidden.shape[1])
        if not self.prefill:
            assert self.attention_q==1 and cache is not None and before==self.previous_kv
            if self.runtime.timing:self.attn_start=torch.cuda.Event(enable_timing=True);self.attn_start.record()
    def attention_end(self,module,args,kwargs,output):
        cache=kwargs.get('past_key_value')
        assert cache is not None, 'KV cache must be enabled'
        after=int(cache.get_seq_length(self.layer))
        assert after==self.attention_before+self.attention_q
        self.previous_kv=after;self.kv_checks+=1
        if not self.prefill and self.runtime.timing:
            end=torch.cuda.Event(enable_timing=True);end.record();self.row['_events']['attention']=(self.attn_start,end)
    def update(self,selected,row):
        r=self.runtime;t=ms();missing,slots=self.cache.plan(selected);self.cache.selected[:]=selected
        row['selection_cache_bookkeeping']=ms()-t
        neurons=self.groups[missing].ravel()
        packed,physical,io=self.reader.read_packed(neurons)
        row.update(ssd_read=io['io_ms'],missing_groups=len(missing),cache_hits=1434-len(missing),
            useful_bytes=io['useful_requested_qweight_bytes'],physical_bytes=io['physical_bytes_read'],read_calls=io['pread_calls'])
        if len(missing):
            n=len(neurons)
            r.staging[:n].copy_(packed)
            with r.event(row,'packed_h2d'):
                r.device_packed[:n].copy_(r.staging[:n],non_blocking=True)
                positions=torch.as_tensor(physical,device='cuda')
            with r.event(row,'gpu_dequant_cache_update'):
                rows=gpu_dequantize(r.device_packed[:n],self.reader.device_scales[positions])
                dest=torch.as_tensor((slots[:,None]*2+np.arange(2)).ravel(),device='cuda')
                self.rows.index_copy_(0,dest,rows)
        elif r.timing:row.update(packed_h2d=0.,gpu_dequant_cache_update=0.)
    def forward(self,x):
        if self.prefill:
            y=self.original(x)
            assert self.seed is not None
            group_scores=self.seed[self.groups].sum(axis=1)
            selected=np.argsort(-group_scores,kind='stable')[:1434].astype(np.int32)
            self.update(selected,dict(_events={}))
            self.cache.commit_importance(group_scores[self.cache.ids])
            # Setup/seed staging may be reused by the next layer, so establish its consumption.
            torch.cuda.current_stream().synchronize()
            self.prefill=False;return y
        row=self.row;t=ms()
        selected=np.concatenate((self.cache.keep,self.fill))
        assert len(self.fill)==215
        if self.runtime.validate:
            assert not np.intersect1d(self.cache.keep,self.fill).size
        self.update(selected,row)
        with self.runtime.event(row,'sparse_mlp'):
            gate=torch.nn.functional.linear(x,self.rows[:,0]);up=torch.nn.functional.linear(x,self.rows[:,1])
            h=torch.nn.functional.silu(gate)*up;y=h@self.rows[:,2]
        with self.runtime.event(row,'importance_keep_update'):
            importance_gpu=h.detach().abs().float().reshape(1434,2).sum(dim=1)
            self.importance_host.copy_(importance_gpu,non_blocking=True)
        # Required CPU consumer completion, outside the flag-driven predictor handoff.
        done=torch.cuda.Event();done.record();done.synchronize()
        importance=self.importance_host.numpy();t=ms()
        if not np.isfinite(importance).all():raise FloatingPointError('Nonfinite sparse activation')
        self.cache.commit_importance(importance)
        # The blocking importance read is outside predictor handoff, and completes GPU input
        # consumption/staging use before the single mapped slot is reused by another layer.
        row['importance_keep_update']=ms()-t;self.calls+=1
        return y
