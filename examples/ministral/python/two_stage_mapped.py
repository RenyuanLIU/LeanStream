"""Two-point prediction and mapped payloads; Python only orchestrates GPU modules.
Prefill uses the existing dense GPTQ path and offline cache seeding. Decode uses
native transactional control, LIVE_PRIORITY pread and mapped-source dequant.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from python.g2_w4a8_runtime import Runtime as OldRuntime,G2MLP,ms
from python.predictor_cpu_native import aligned
from python.v1_storage import PackedReader,gpu_dequantize


def ptr(a):return a.ctypes.data

def capsule(pointer):
    f=C.pythonapi.PyCapsule_New;f.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];f.restype=C.py_object
    return torch.utils.dlpack.from_dlpack(f(pointer,b'dltensor',None))

class Native:
    def __init__(self,library,predictor,parent=None,group_size=2,stable=False,model_spec=None,predictor_stride=1024):
        l=self.lib=C.CDLL(str(Path(library).resolve()));p=C.c_void_p;i=C.c_int;u=C.c_uint64
        signatures={
            'lp_create':([C.c_char_p],p),'lp_destroy':([p],None),'lp_error':([],C.c_char_p),
            'lp_mapped':([p],p),'lp_shared':([p],p),'lp_cache':([p,i],p),'lp_times':([p],C.POINTER(C.c_double)),
            'lp_fill':([p,i],p),'lp_bind':([p,i,i,p,p,p,p,p,p],i),'lp_reset':([p],i),
            'lp_p1_policy':([p,i],i),'lp_p1':([p,i,i],i),'lp_p2':([p,p,p,C.c_float,p],i),'lp_update':([p,p],i),'lp_keep':([p],i),
            'lm_hidden':([p,p,i,p],i),'lm_wait':([p,i],i),'lm_importance':([p,p,p],i),
            'lm_pointer':([p,i,i],p),'lm_hidden_dlpack':([p,i],p),'si_dlpack':([p],p),
            'si_profile_events':([p,p,p],None),'lc_cache_plan_selected':([p,p],i),
            'lc_cache_mark_launched':([p,u],i),'lc_cache_commit':([p,u],i),
            'lc_cache_ticket':([p],u),'lc_cache_update_keep':([p,p],i),'lc_cache_view':([p,i],p)}
        for name,(args,result) in signatures.items():f=getattr(l,name);f.argtypes=args;f.restype=result
        if parent is None:self.ctx=l.lp_create(str(predictor).encode())
        else:
            l.lp_create_bank.argtypes=[p];l.lp_create_bank.restype=p;self.ctx=l.lp_create_bank(parent.ctx)
        assert self.ctx,l.lp_error()
        if model_spec is not None:
            self.group_size=2;self.active=model_spec['active'];self.core=model_spec['keep'];self.fill=self.active-self.core
            l.lp_model_config.argtypes=[p,i,i,i,i];l.lp_model_config.restype=i
            self.check(l.lp_model_config(self.ctx,model_spec['H'],model_spec['I'],model_spec['layers'],int(stable)))
            if predictor_stride != 1024:
                l.lp_predictor_stride.argtypes=[p,i];l.lp_predictor_stride.restype=i
                self.check(l.lp_predictor_stride(self.ctx,predictor_stride))
        else:
            self.group_size=group_size;self.active=2868//group_size;self.core={2:1219,3:812,4:609}[group_size];self.fill=self.active-self.core
        if model_spec is None and (group_size!=2 or stable):
            l.lp_group_config.argtypes=[p,i,i];l.lp_group_config.restype=i
            self.check(l.lp_group_config(self.ctx,group_size,int(stable)))
        self.mapped=l.lp_mapped(self.ctx);self.shared=l.lp_shared(self.ctx)
        self.hidden=[capsule(l.lm_hidden_dlpack(self.mapped,s)) for s in (0,1)]
        self.rms=capsule(l.si_dlpack(self.shared))
        self.times=np.ctypeslib.as_array(l.lp_times(self.ctx),shape=(24,))
    def check(self,status):
        if status:raise RuntimeError(self.lib.lp_error().decode()+f' (native status {status})')
    def view(self,cache,kind,n):
        return np.ctypeslib.as_array((C.c_int32*n).from_address(self.lib.lc_cache_view(cache,kind)))

class Runtime(OldRuntime):
    def __new__(cls,model,root,predictor_library,library,timing=False,validate=False,split=False,legacy_split=False,**options):
        if cls is Runtime and split and not legacy_split:
            from python.split_async import Runtime as AsyncRuntime
            return object.__new__(AsyncRuntime)
        return object.__new__(cls)
    def __init__(self,model,root,predictor_library,library,timing=False,validate=False,split=False,legacy_split=False):
        self.model=model;self.root=Path(root);self.timing=timing;self.validate=validate
        self.predictor_library=predictor_library
        self.records=[];self.layers=[];self.native=Native(library,predictor_library)
        self.split=None
        if split:
            from python.triton_split_runtime import SplitEngine
            self.split=SplitEngine(self.native)
        for i,layer in enumerate(model.model.layers):
            if type(layer.self_attn).__name__!='MistralSdpaAttention':raise ValueError('HF SDPA required')
            m=MappedMLP(self,layer.mlp,i);self.layers.append(m)
            layer.mlp=m;layer.post_attention_layernorm=MappedNorm(layer.post_attention_layernorm,m)
            layer.register_forward_pre_hook(m.layer_begin)
            layer.register_forward_hook(m.layer_end)
            layer.self_attn.register_forward_pre_hook(m.attention_begin,with_kwargs=True)
            layer.self_attn.register_forward_hook(m.attention_end,with_kwargs=True)
        self.reset()
        if self.split:self.split.warm(self.layers[0])
    def reset(self):
        self.native.check(self.native.lib.lp_reset(self.native.ctx))
        for m in self.layers:m.reset()
    def resolve(self):
        for row in self.records:
            for key,(a,b) in row.pop('_events',{}).items():row[key]=a.elapsed_time(b)
        return self.records

class MappedNorm(nn.Module):
    def __init__(self,original,mlp):
        super().__init__();self.original=original;object.__setattr__(self,'mlp',mlp)
    def forward(self,x):
        m=self.mlp;r=m.runtime;n=m.native;l=n.lib
        if m.prefill:return self.original(x)
        if x.shape!=(1,1,m.native.hidden[0].shape[-1]):raise ValueError('Decode must be one token')
        if r.timing:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();b.record();l.si_profile_events(n.shared,a.cuda_event,b.cuda_event)
            m.row['_events']['post_attention_norm']=(a,b)
        if r.split:
            r.split.enqueue(m,x,self.original)
        else:
            n.check(l.lp_p2(n.ctx,x.data_ptr(),self.original.weight.data_ptr(),self.original.variance_epsilon,torch.cuda.current_stream().cuda_stream))
        if r.validate and not m.input_checked:
            # Validation only; old single-predictor arithmetic on exactly the same mapped RMS input.
            from python.predictor_cpu_native import Kernel
            expected=self.original(x)
            if getattr(r,'numerical_warnings',None) is not None:
                from python.finalist_gate import check_rms
                check_rms(n.rms,expected,r.numerical_warnings,m.layer)
            else:
                torch.testing.assert_close(n.rms,expected,rtol=1e-3,atol=1e-3)
            full=n.rms.cpu().numpy().reshape(-1);selected=full[m.features].astype(np.float32)
            scale=np.float32(np.max(np.abs(selected))/np.float32(127));scale=np.float32(1) if scale==0 else scale
            quant=aligned(np.zeros(1664,np.int8));quant[:1638]=np.clip(np.rint(selected/scale),-127,127).astype(np.int8)
            kernel=Kernel(r.predictor_library,4,12,rows=getattr(m,"predictor_rows",7168),k=n.fill)
            keep=n.view(m.cache,2,n.core)
            if n.group_size==3:keep=np.append(keep,np.int32(4778)).astype(np.int32)
            fill=kernel.predict(m.weights,m.scales,quant,scale,keep,skip=False)
            actual=np.ctypeslib.as_array((C.c_int32*n.fill).from_address(l.lp_fill(n.ctx,1)))
            if not getattr(r,'relaxed_p2_m',0):
                np.testing.assert_array_equal(fill,actual)
            kernel.close()
            capture=getattr(r,"capture_inputs",None)
            if capture is not None:
                np.savez(capture/("layer_%02d.npz"%m.layer),p1=n.hidden[0].cpu().numpy().reshape(4096),p2=full,keep=n.view(m.cache,2,n.core).copy(),group_size=n.group_size)
            m.input_checked=True
        return n.rms

class MappedMLP(G2MLP):
    def __init__(self,runtime,original,layer):
        nn.Module.__init__(self);self.original=original;object.__setattr__(self,'runtime',runtime)
        self.layer=layer;r=runtime;n=r.native;self.native=n;l=n.lib
        new_model=getattr(r,'model_spec',None) is not None
        folder=(r.predictor_root/f'layer_{layer:02d}'/f'd{r.predictor_dim}') if new_model else (r.root/'predictor'/f'layer_{layer:02d}')
        with np.load(folder/(('runtime.npz' if (folder/'runtime.npz').exists() else 'predictor.npz') if new_model else 'mapping.npz')) as z:
            self.groups=z['groups']
            self.features=aligned(z['features'].astype(np.int32)) if 'features' in z else None
            self.projection=aligned(z['projection'].astype(np.float32)) if 'projection' in z else None
            if new_model:assert (self.features is None)!=(self.projection is None)
        self.weights=aligned(np.fromfile(folder/'w4.bin',np.uint8));self.scales=aligned(np.fromfile(folder/'scales4.bin',np.float32))
        stride=r.predictor_dim if new_model else 1664
        assert (self.weights.size==len(self.groups)*stride and self.scales.size==len(self.groups)*2) if new_model else (self.weights.size==7168*2*832 and self.scales.size==7168*2)
        self.reader=PackedReader(r.root/'g2_weights'/f'layer_{layer:02d}',0,2)
        assert np.array_equal(self.reader.mapping[self.groups.ravel()],np.arange(self.reader.meta['I']))
        self.reader.scales_on(torch.device('cuda'))
        h=self.reader.meta['H'];group_bytes=self.reader.meta['bytes_per_bundle']
        self.rows=None if runtime.split else torch.empty((n.active*2,3,h),dtype=torch.float16,device='cuda')
        self.packed_cache=torch.empty((n.active*group_bytes,),dtype=torch.uint8,device='cuda') if runtime.split else None
        offsets=np.ascontiguousarray(self.reader.mapping[self.groups[:,0]]*(3*h//2),dtype=np.int64)
        n.check(l.lp_bind(n.ctx,layer,self.reader.fd,ptr(offsets),ptr(self.features) if self.features is not None else 0,ptr(self.weights),ptr(self.scales),self.reader.device_scales.data_ptr(),self.rows.data_ptr() if self.rows is not None else 0))
        if new_model and getattr(r,'spec_p1_size',0):
            if r.predictor_dim!=1664 or self.features is None:
                raise ValueError('Cheap speculation requires the feature-gather predictor')
            hot_path=folder/'speculative_hot.npy'
            meta=json.loads((folder/'speculative_hot.json').read_text())
            predictor_meta=json.loads((folder/'manifest.json').read_text())
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            if (meta['layer']!=layer or meta['source_sha256']!=predictor_meta['identity']['source_sha256']
                    or meta['predictor_manifest_sha256']!=sha(folder/'manifest.json')
                    or meta['order_sha256']!=sha(hot_path)):
                raise ValueError('Speculative candidate identity mismatch')
            self.spec_hot=aligned(np.load(hot_path).astype(np.int32))
            if len(self.spec_hot)!=len(self.groups) or len(np.unique(self.spec_hot))!=len(self.groups):
                raise ValueError('Speculative candidate order invalid')
            l.lp_speculative_bind.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int]
            l.lp_speculative_bind.restype=C.c_int
            n.check(l.lp_speculative_bind(n.ctx,layer,ptr(self.spec_hot),len(self.spec_hot)))
        if new_model and getattr(r,'p1_lite_budget',0):
            lite_folder=r.p1_lite_root/f'layer_{layer:02d}'/f'd{r.p1_lite_features}'
            lite_meta=json.loads((lite_folder/'manifest.json').read_text())
            full_meta=json.loads((folder/'manifest.json').read_text())
            digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            if (lite_meta['layer']!=layer or lite_meta['checkpoint']!=full_meta['identity']['checkpoint'] or
                    lite_meta['source_sha256']!=full_meta['identity']['source_sha256'] or
                    lite_meta['rows']!=len(self.groups) or
                    lite_meta['active_features']!=r.p1_lite_features or
                    lite_meta['padded_stride']!=r.p1_lite_features or
                    any(lite_meta['files'][name]!=digest(lite_folder/name)
                        for name in ('runtime.npz','w4.bin','scales4.bin'))):
                raise ValueError('Compact P1 artifact identity mismatch')
            with np.load(lite_folder/'runtime.npz') as data:
                self.lite_features=aligned(data['features'].astype(np.int32))
            self.lite_weights=aligned(np.fromfile(lite_folder/'w4.bin',np.uint8))
            self.lite_scales=aligned(np.fromfile(lite_folder/'scales4.bin',np.float32))
            if (self.lite_features.size!=r.p1_lite_features or
                    self.lite_weights.size!=len(self.groups)*r.p1_lite_features or
                    self.lite_scales.size!=len(self.groups)*2):
                raise ValueError('Compact P1 artifact shape mismatch')
            l.lp_lite_bind.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
            l.lp_lite_bind.restype=C.c_int
            n.check(l.lp_lite_bind(n.ctx,layer,ptr(self.lite_features),ptr(self.lite_weights),ptr(self.lite_scales)))
        if new_model and getattr(r,'mid_partial_root',None) is not None:
            if r.predictor_dim!=1664 or getattr(r,'spec_mid_size',0)==0:
                raise ValueError('Partial-head midpoint requires native feature predictor and candidate midpoint')
            tiny_features=getattr(r,'tiny_mid_features',0)
            mid_folder=r.mid_partial_root/f'layer_{layer:02d}'
            if tiny_features:mid_folder=mid_folder/f'd{tiny_features}'
            mid_meta=json.loads((mid_folder/'manifest.json').read_text())
            source_meta=json.loads((folder/'manifest.json').read_text())
            digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            if (mid_meta['layer']!=layer or mid_meta['checkpoint']!=source_meta['identity']['checkpoint'] or
                    mid_meta['rows']!=len(self.groups) or
                    mid_meta['native_stride']!=(tiny_features or 1664) or
                    any(mid_meta['files'][name]!=digest(mid_folder/name)
                        for name in ('runtime.npz','w4.bin','scales4.bin')) or
                    (not tiny_features and mid_meta['source_predictor_manifest_sha256']!=digest(folder/'manifest.json')) or
                    (tiny_features and mid_meta['active_features']!=tiny_features)):
                raise ValueError('Partial-head midpoint artifact identity mismatch')
            if tiny_features:
                parent=r.mid_partial_root.parent/'midpoint_partial'/f'layer_{layer:02d}'/'manifest.json'
                if mid_meta['parent_manifest_sha256']!=digest(parent):
                    raise ValueError('Tiny Mid parent identity mismatch')
            with np.load(mid_folder/'runtime.npz') as mid_data:
                self.mid_features=aligned(mid_data['features'].astype(np.int32))
            self.mid_weights=aligned(np.fromfile(mid_folder/'w4.bin',np.uint8))
            self.mid_scales=aligned(np.fromfile(mid_folder/'scales4.bin',np.float32))
            if (self.mid_features.size!=(tiny_features or 1638) or
                    self.mid_weights.size!=len(self.groups)*(tiny_features or 1664) or
                    self.mid_scales.size!=len(self.groups)*2 or
                    not np.all((self.mid_features>=0)&(self.mid_features<r.model_spec['H']))):
                raise ValueError('Partial-head midpoint artifact geometry mismatch')
            l.lp_mid_partial_bind.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
            l.lp_mid_partial_bind.restype=C.c_int
            n.check(l.lp_mid_partial_bind(n.ctx,layer,ptr(self.mid_features),ptr(self.mid_weights),ptr(self.mid_scales)))
        if new_model and self.projection is not None:
            l.lp_bind_projection.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int];l.lp_bind_projection.restype=C.c_int
            n.check(l.lp_bind_projection(n.ctx,layer,ptr(self.projection),r.predictor_dim))
        self.cache=l.lp_cache(n.ctx,layer);self.input_checked=False;self.kv_checks=0
        self.hidden_checked=False;original.down_proj.register_forward_pre_hook(self.capture)
    def reset(self):self.prefill=True;self.seed=None;self.previous_kv=None;self.calls=0
    def layer_begin(self,module,inputs):
        self.was_prefill=self.prefill
        if self.prefill:return
        self.start=ms();self.row=dict(layer=self.layer,token=self.calls,_events={})
        r=self.runtime;n=self.native;l=n.lib;s=self.layer%2;stream=torch.cuda.current_stream().cuda_stream
        if self.layer==0:
            with r.event(self.row,'mapped_hidden_input'):n.check(l.lm_hidden(n.mapped,inputs[0].data_ptr(),s,stream))
        n.check(l.lm_wait(n.mapped,s))
        if r.validate and not self.hidden_checked:
            torch.testing.assert_close(n.hidden[s],inputs[0],rtol=0,atol=0);self.hidden_checked=True
        n.check(l.lp_p1(n.ctx,self.layer,s))
        return (n.hidden[s],)+inputs[1:]
    def layer_end(self,module,inputs,output):
        if self.was_prefill:return
        r=self.runtime;n=self.native;l=n.lib;s=(self.layer+1)%2;stream=torch.cuda.current_stream().cuda_stream
        with r.event(self.row,'mapped_hidden_output'):n.check(l.lm_hidden(n.mapped,output[0].data_ptr(),s,stream))
        # Actual layer boundary completion via epoch, not a CUDA blocking API.
        n.check(l.lm_wait(n.mapped,s))
        if r.validate and not getattr(self,'output_checked',False):
            torch.testing.assert_close(n.hidden[s],output[0],rtol=0,atol=0);self.output_checked=True
        self.row['layer_total']=ms()-self.start
        if r.timing:r.records.append(self.row)
        return (n.hidden[s],)+output[1:]
    def attention_begin(self,module,args,kwargs):
        super().attention_begin(module,args,kwargs)
        if self.runtime.validate and not self.prefill and self.layer in (0,15,31) and not getattr(self,'attention_checked',False):
            from copy import copy
            cache=kwargs['past_key_value'];ref=copy(cache)
            ref.key_cache=list(cache.key_cache);ref.value_cache=list(cache.value_cache)
            ref.key_cache[self.layer]=cache.key_cache[self.layer].clone()
            ref.value_cache[self.layer]=cache.value_cache[self.layer].clone()
            self.ref_kwargs=dict(kwargs,past_key_value=ref)
            self.ref_args=args
            if 'hidden_states' in kwargs:self.ref_kwargs['hidden_states']=kwargs['hidden_states'].clone()
            elif args:self.ref_args=(args[0].clone(),)+args[1:]
    def attention_end(self,module,args,kwargs,output):
        super().attention_end(module,args,kwargs,output)
        if hasattr(self,'ref_kwargs'):
            # Validation only: bypass hooks, reuse unmodified installed HF SDPA with an independent KV snapshot.
            expected=module.forward(*self.ref_args,**self.ref_kwargs)
            torch.testing.assert_close(output[0],expected[0],rtol=1e-3,atol=1e-3)
            actual_cache=kwargs['past_key_value'];ref=self.ref_kwargs['past_key_value']
            torch.testing.assert_close(actual_cache.key_cache[self.layer],ref.key_cache[self.layer],rtol=0,atol=0)
            torch.testing.assert_close(actual_cache.value_cache[self.layer],ref.value_cache[self.layer],rtol=0,atol=0)
            del self.ref_args;del self.ref_kwargs;self.attention_checked=True
    def seed_cache(self):
        # Prefill-only preparation: existing dense prefill and FP16 sparse seed. No streaming claim.
        n=self.native;l=n.lib
        scores=self.seed[self.groups].sum(1);selected=np.argsort(-scores,kind='stable')[:n.active].astype(np.int32)
        n.check(l.lc_cache_plan_selected(self.cache,ptr(selected)))
        packed,physical,_=self.reader.read_packed(self.groups[selected].ravel())
        if self.runtime.split:
            self.packed_cache.copy_(packed.reshape(-1)) # Prefill-only cache initialization.
        else:
            actual=gpu_dequantize(packed.cuda(),torch.from_numpy(self.reader.scales[physical]).cuda())
            self.rows.copy_(actual)
        torch.cuda.current_stream().synchronize() # Prefill only.
        ticket=l.lc_cache_ticket(self.cache);n.check(l.lc_cache_mark_launched(self.cache,ticket));n.check(l.lc_cache_commit(self.cache,ticket))
        importance=np.ascontiguousarray(scores[selected],dtype=np.float32)
        n.check(l.lc_cache_update_keep(self.cache,ptr(importance)))
    def forward(self,x):
        if self.prefill:
            y=self.original(x);assert self.seed is not None;self.seed_cache();self.prefill=False;return y
        r=self.runtime;n=self.native;l=n.lib;stream=torch.cuda.current_stream().cuda_stream
        if r.split:
            y=r.split.finish(self);self.calls+=1;return y
        with r.event(self.row,'mapped_dequant'):n.check(l.lp_update(n.ctx,stream))
        with r.event(self.row,'sparse_mlp'):
            gate=torch.nn.functional.linear(x,self.rows[:,0]);up=torch.nn.functional.linear(x,self.rows[:,1])
            h=torch.nn.functional.silu(gate)*up;y=h@self.rows[:,2]
        with r.event(self.row,'mapped_importance'):n.check(l.lm_importance(n.mapped,h.data_ptr(),stream))
        n.check(l.lp_keep(n.ctx))
        names=('p1_quant','p1_w4a8','p1_total','p1_publish','p2_quant','p2_w4a8','p2_total','p2_publish',
               'native_plan','mlp_io_wait','native_keep','p1_requested','loaded_before_p2','ready_at_p2',
               'final_missing','fill_overlap','speculative_waste','pread_calls','rms_ready_wait',
               'p2_update_active','p1_cache_prefetch','p1_pio_stats','p1_pio_update','p1_wait_started')
        for i,key in enumerate(names):self.row[key]=float(n.times[i])
        self.calls+=1;return y
