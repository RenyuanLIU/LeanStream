"""Production banked split maintenance. Device output does not depend on CPU keep.
Each layer owns its staging/descriptors until the next token; no next-layer
bank reuse dependency. The default split=False runtime is unchanged.
"""
from contextlib import contextmanager
import ctypes as C
import os
import time
import torch
import triton
from pathlib import Path
from python.two_stage_mapped import Runtime as BaseRuntime,Native,MappedMLP,MappedNorm,capsule
from python.triton_split_runtime import SplitEngine
from python.triton_split_mlp import reduce_split,publish_next_hidden

class Engine(SplitEngine):
    def __init__(self,n,no_waste=False):
        super().__init__(n,no_waste=no_waste)
        self.next_native=None;self.next_hidden=None;self.next_flag=None
        self.b_done=torch.cuda.Event(enable_timing=False)
        self.b_done.record() # Force lazy CUDA handle creation during initialization, never decode.
        self.input_ready=torch.cuda.Event(enable_timing=False)
        self.input_ready.record()
        self.input_done=torch.cuda.Event(enable_timing=False)
        self.input_done.record()
        p=C.c_void_p;i=C.c_int
        for name,args,ret in (
            ('lm_hidden_reserve',[p],C.c_uint32),('lm_hidden_flag_dlpack',[p],p),
            ('lm_hidden_nowait',[p,p,i,p],i),('lm_importance_nowait',[p,p,p],i),
            ('lp_speculation',[p,i],i),('lp_final_qd',[p,i],i),('lp_idle_before_p2',[p],C.c_double),
            ('lp_async_enable',[p],i),('lp_control_submit',[p,i,i],i),('lp_control_arm',[p,i,i],i),
            ('lp_ready_scheduler_enable',[p],i),('lp_ready_notification_mode',[p,i],i),
            ('lp_ready_predictor_trace',[p,i],p),
            ('lp_ready_p1_io_trace',[p],p),('lp_control_wait',[p],i),
            ('lp_prepare_a',[p,i],i),('lp_maintenance_wait',[p,i],i),
            ('lp_maintenance_submit',[p],i),('lp_main_waits',[p],C.c_uint64),
            ('lp_p0_start',[p,i],i),
            ('lp_main_wait_ms',[p],C.c_double)):
            f=getattr(n.lib,name);f.argtypes=args;f.restype=ret
        if self.no_waste:
            n.lib.lm_importance_no_waste_nowait.argtypes=[p,p,p,p]
            n.lib.lm_importance_no_waste_nowait.restype=i
        if getattr(self,'public_mode','old')!='old':
            self.public_callback=C.CFUNCTYPE(C.c_int,C.c_void_p)(self._public_compute)
            n.lib.lp_public_compute_bind.argtypes=[p,p,p,p];n.lib.lp_public_compute_bind.restype=i
            n.lib.lp_public_enqueue.argtypes=[p,p,p,p];n.lib.lp_public_enqueue.restype=i
            n.check(n.lib.lp_public_compute_bind(n.ctx,C.cast(self.public_callback,p),None,None))
            n.lib.lp_public_reduce_bind.argtypes=[p,p,i];n.lib.lp_public_reduce_bind.restype=i
            n.check(n.lib.lp_public_reduce_bind(n.ctx,self.reduced.host.ctypes.data,self.width//32))
            self.public_mlp=None
    def _public_compute(self,unused):
        try:
            m=self.public_mlp;self.launch(m,self.epoch)
            # Compatibility adapter controls predictor placement; core API has
            # no predictor concept. Same arm point: after A/B enqueue.
            n=self.n;n.check((n.lib.lp_control_arm if m.runtime.ready_scheduler else n.lib.lp_control_submit)(n.ctx,2,0))
            self.b_done.record();self._public_reduce(m);return 0
        except Exception as error:
            self.public_error=error;return 5
    def _public_reduce(self,m):
        publish=m.runtime.producer_publication and self.next_native is not None
        ordered=publish and m.runtime.publication_method=='ordered';epoch=0
        if publish:
            epoch=self.next_native.lib.lm_hidden_reserve(self.next_native.mapped)
            if not epoch:raise RuntimeError('Next-layer hidden bank not reusable')
        reducer=reduce_split;programs=self.programs;extra={}
        if self.public_mode=='progressive':
            from kernels.compact_reduction import reduce_split as reducer
            programs=self.pa+(self.pb+1)//2;extra={'CORE':self.pa}
        if publish and not ordered:
            self.public_reduce_kernel=reducer[(self.width//32,)](self.partial,self.y,self.reduced.tensor,self.epoch,self.programs,
                triton.next_power_of_2(programs),32,True,self.final_residual,self.next_hidden,
                self.next_flag,epoch,self.next_native.notify_bell,
                NOTIFY=self.next_native.notify_bell is not None,WIDTH=self.width,num_warps=4,num_stages=1,**extra)
        else:
            self.public_reduce_kernel=reducer[(self.width//32,)](self.partial,self.y,self.reduced.tensor,self.epoch,self.programs,
                triton.next_power_of_2(programs),32,WIDTH=self.width,num_warps=4,num_stages=1,**extra)
        if ordered:
            publish_next_hidden[(1,)](self.y,self.final_residual,self.next_hidden,self.next_flag,epoch,
                self.next_native.notify_bell,NOTIFY=self.next_native.notify_bell is not None,
                WIDTH=self.width,num_warps=4,num_stages=1)
    def enqueue(self,m,residual,norm):
        host_begin=time.perf_counter_ns()
        self.final_residual=residual
        n=self.n
        self.epoch=n.lib.lp_split_start(n.ctx,residual.data_ptr(),norm.weight.data_ptr(),norm.variance_epsilon,
                                     torch.cuda.current_stream().cuda_stream)
        if not self.epoch:raise RuntimeError(n.lib.lp_error().decode())
        if self.no_waste and m.runtime.validate:self.current_a=self.a.host.copy()
        if getattr(m.runtime,'capture_first',False) and m.layer==0 and m.calls==0:
            self.capture_a=self.a.host.copy()  # Offline gate: before next-token KEEP prebuild can replace it.
        checking=not self.no_waste and m.runtime.validate and (getattr(m.runtime,'reference_all_layers',False) or
            (m.layer in (0,15,31) and not getattr(m,'split_checked',False)))
        if checking:
            before=n.view(m.cache,0,n.active).copy();keep=n.view(m.cache,2,n.core).copy()
            self.current_a=self.a.host.copy()
        # Do not allow readiness-polling B CTAs to starve the side-stream
        # producer that P1 depends on. Normally completed during attention.
        if not m.runtime.producer_publication or m.layer==0:
            torch.cuda.current_stream().wait_event(self.input_done)
        if self.execution and not self.ready_ids:
            self.launch_keep(m,self.epoch)
            n.check((n.lib.lp_control_arm if m.runtime.ready_scheduler else n.lib.lp_control_submit)(n.ctx,2,0))
            self.launch_dynamic(m,self.epoch)
        else:
            if getattr(self,'public_mode','old')!='old':
                self.public_mlp=m
                if getattr(self,"native_cuda_bound",False) and self.public_exec_counts is not None:self.public_exec_counts.host.fill(0)
                n.check(n.lib.lp_public_enqueue(n.ctx,n.rms.data_ptr(),self.y.data_ptr(),torch.cuda.current_stream().cuda_stream))
            else:
                self.launch(m,self.epoch)
                n.check((n.lib.lp_control_arm if m.runtime.ready_scheduler else n.lib.lp_control_submit)(n.ctx,2,0))
        if m.runtime.validate:
            n.check(n.lib.lp_control_wait(n.ctx)) # Offline validation only.
        if checking:
            import numpy as np
            from python.predictor_cpu_native import Kernel,aligned
            full=n.hidden[0].cpu().numpy().reshape(-1);x=full[m.features].astype(np.float32)
            scale=np.float32(np.max(np.abs(x))/np.float32(127));scale=np.float32(1) if scale==0 else scale
            quant=aligned(np.zeros(1664,np.int8));quant[:1638]=np.clip(np.rint(x/scale),-127,127).astype(np.int8)
            reference=Kernel(m.runtime.predictor_library,4,12,rows=getattr(m,"predictor_rows",7168),k=n.fill)
            excluded=keep if n.group_size!=3 else np.append(keep,np.int32(4778)).astype(np.int32)
            expected=reference.predict(m.weights,m.scales,quant,scale,excluded,skip=False).copy();reference.close()
            if not m.runtime.p1_lite_budget:
                p1=np.ctypeslib.as_array((C.c_int32*n.fill).from_address(n.lib.lp_fill(n.ctx,0)))
                np.testing.assert_array_equal(p1,expected)
            fill=np.ctypeslib.as_array((C.c_int32*n.fill).from_address(n.lib.lp_fill(n.ctx,1))).copy()
            if self.execution:
                # Offline exact native P2 check for this invocation's RMS input.
                x=n.rms.cpu().numpy().reshape(-1)[m.features].astype(np.float32)
                scale=np.float32(np.max(np.abs(x))/np.float32(127));scale=np.float32(1) if scale==0 else scale
                quant.fill(0);quant[:1638]=np.clip(np.rint(x/scale),-127,127).astype(np.int8)
                reference=Kernel(m.runtime.predictor_library,4,12,rows=getattr(m,'predictor_rows',7168),k=n.fill)
                expected2=reference.predict(m.weights,m.scales,quant,scale,excluded,skip=False,debug=True).copy()
                if m.runtime.relaxed_p2_m:
                    from types import SimpleNamespace
                    raise NotImplementedError("Optional model adapter supports REPAIRED_KNEE without relaxed P2")
                    group_of=np.argsort(m.groups.ravel())//2
                    resident_neurons=m.groups[before].ravel()
                    state=SimpleNamespace(groups=m.groups,g=2,group_of=group_of,
                        fill_groups=n.fill,slots={int(neuron):i for i,neuron in enumerate(resident_neurons)})
                    fill_neurons,_=relaxed_groups(state,reference.debug[2],m.groups[keep].ravel(),
                        m.runtime.relaxed_p2_m,m.runtime.relaxed_p2_p)
                    expected2=group_of[fill_neurons[::2]].astype(np.int32)
                reference.close()
                np.testing.assert_array_equal(fill,expected2)
                self.current_fill=fill.copy()
            selected=np.concatenate((keep,fill));wanted=set(selected.tolist());cached=set(before.tolist())
            missing=np.array([g for g in selected if g not in cached],np.int32)
            slots=np.array([i for i,g in enumerate(before) if g not in wanted],np.int32)
            np.testing.assert_array_equal(n.view(m.cache,4,len(missing)),missing)
            np.testing.assert_array_equal(n.view(m.cache,5,len(slots)),slots)
            np.testing.assert_array_equal(n.view(m.cache,0,n.active),before)
            np.testing.assert_array_equal(self.a.host.reshape(-1,5)[:,0],keep)
        self.host_enqueue_ms=(time.perf_counter_ns()-host_begin)/1e6
    def finish(self,m):
        host_begin=time.perf_counter_ns()
        n=self.n
        if getattr(self,'public_mode','old')=='old':
            self.b_done.record()
            publish=m.runtime.producer_publication and self.next_native is not None
            ordered=publish and m.runtime.publication_method=='ordered'
            if publish:
                epoch=self.next_native.lib.lm_hidden_reserve(self.next_native.mapped)
                if not epoch:raise RuntimeError('Next-layer hidden bank not reusable')
            reducer=reduce_split;reduction_programs=self.programs;extra={}
            if getattr(self,'public_mode','old')=='progressive':
                from kernels.compact_reduction import reduce_split as reducer
                reduction_programs=self.pa+(self.pb+1)//2;extra={'CORE':self.pa}
            with m.runtime.event(m.row,"reduction"):
                if publish and not ordered:
                    reducer[(self.width//32,)](self.partial,self.y,self.reduced.tensor,self.epoch,self.programs,
                        triton.next_power_of_2(reduction_programs),32,True,self.final_residual,self.next_hidden,
                        self.next_flag,epoch,self.next_native.notify_bell,
                        NOTIFY=self.next_native.notify_bell is not None,WIDTH=self.width,num_warps=4,num_stages=1,**extra)
                else:
                    reducer[(self.width//32,)](self.partial,self.y,self.reduced.tensor,self.epoch,self.programs,
                        triton.next_power_of_2(reduction_programs),32,WIDTH=self.width,num_warps=4,num_stages=1,**extra)
            with m.runtime.event(m.row,"publication"):
                if ordered:
                    publish_next_hidden[(1,)](self.y,self.final_residual,self.next_hidden,self.next_flag,epoch,
                        self.next_native.notify_bell,NOTIFY=self.next_native.notify_bell is not None,
                        WIDTH=self.width,num_warps=4,num_stages=1)
        # This stream waits only on B. Neither reduction nor following attention
        # depends on packed-cache maintenance or CPU ownership accounting.
        with torch.cuda.stream(m.runtime.maintenance):
            m.runtime.maintenance.wait_event(self.b_done)
            stream=m.runtime.maintenance.cuda_stream
            if m.runtime.diagnostic:
                with m.runtime.event(m.row,'packed_cache_update'):
                    n.check(n.lib.lp_split_update(n.ctx,m.packed_cache.data_ptr(),stream))
                with m.runtime.event(m.row,'mapped_importance'):
                    if self.no_waste:
                        n.check(n.lib.lm_importance_no_waste_nowait(n.mapped,self.h.data_ptr(),
                            self.ready.tensor[2:3].data_ptr(),stream))
                    else:n.check(n.lib.lm_importance_nowait(n.mapped,self.h.data_ptr(),stream))
            else:
                n.check(n.lib.lp_split_update(n.ctx,m.packed_cache.data_ptr(),stream))
                if self.no_waste:
                    n.check(n.lib.lm_importance_no_waste_nowait(n.mapped,self.h.data_ptr(),
                        self.ready.tensor[2:3].data_ptr(),stream))
                else:n.check(n.lib.lm_importance_nowait(n.mapped,self.h.data_ptr(),stream))
        gate_this_layer=m.runtime.validate and __import__('os').getenv('LEANSTREAM_NW_SKIP_VALIDATION') != '1' and (
            (getattr(m.runtime,'reference_all_layers',False) and
             (not self.no_waste or m.calls==0)) or
            (m.layer in (0,15,31) and not getattr(m,'split_checked',False)))
        no_waste_checking=self.no_waste and gate_this_layer
        if no_waste_checking:
            # Offline gate only. Current cache/staging sources still own the
            # exact EXEC descriptors; late eviction has not overwritten them.
            torch.cuda.synchronize()
            self.validate_no_waste_compute(m)
        # Diagnostic only: isolate the effect of waiting for all queued GPU
        # consumers before allowing maintenance to advance. Never enabled in
        # clean timing or production execution.
        if self.no_waste and __import__('os').getenv('LEANSTREAM_NW_SYNC_PROBE') == '1':
            torch.cuda.synchronize()
        n.check(n.lib.lp_maintenance_submit(n.ctx))
        if self.no_waste and __import__('os').getenv('LEANSTREAM_NW_WAIT_MAINTENANCE') == '1':
            n.check(n.lib.lp_maintenance_wait(n.ctx,0))
        if gate_this_layer:
            n.check(n.lib.lp_maintenance_wait(n.ctx,0))
            if self.no_waste:
                self.validate_no_waste_retention(m)
                m.split_checked=True
                return self.y
            next_a=self.a.host.copy();self.a.host[:]=self.current_a
            self.validate(m)
            if self.execution:
                import numpy as np
                np.testing.assert_array_equal(np.sort(self.b.host.reshape(-1,5)[:,0]),np.sort(self.current_fill))
            self.a.host[:]=next_a;m.split_checked=True
        self.host_finish_ms=(time.perf_counter_ns()-host_begin)/1e6
        return self.y

    def validate_no_waste_compute(self,m):
        from python.v1_storage import gpu_dequantize
        import numpy as np
        n=self.n;optional=int(self.ready.host[1]);count=int(self.ready.host[2])
        if not (n.fill<=optional<=self.max_optional and count==n.core+optional):
            raise AssertionError('No-waste descriptor count invalid')
        a=getattr(self,'current_a',self.a.host)
        b=self.public_desc.host if self.public_mode=='progressive' else self.b.host
        desc=np.concatenate((a.reshape(-1,5),b.reshape(-1,5)[:optional])).copy()
        groups=desc[:,0];dest=desc[:,3];kind=desc[:,4];slots=desc[:,2]
        if len(np.unique(groups))!=count or not np.array_equal(np.sort(dest),np.arange(count)):
            raise AssertionError('No-waste EXEC group/destination duplicate')
        if not np.isin(kind,[0,1]).all() or (slots<0).any():
            raise AssertionError('No-waste invalid source kind/slot')
        physical=m.reader.mapping[m.groups[groups].ravel()]
        if not np.array_equal(desc[:,1],physical.reshape(-1,2)[:,0]//2):
            raise AssertionError('No-waste logical/physical scale mapping')
        cache=m.packed_cache.reshape(n.active,-1)
        staging=self.staging.reshape(n.n_groups if hasattr(n,'n_groups') else 7168,-1)
        if cache.shape[1]!=staging.shape[1]:raise AssertionError('Packed source width changed')
        packed=torch.empty((count,cache.shape[1]),dtype=cache.dtype,device=cache.device)
        for source,bank in ((0,cache),(1,staging)):
            indices=np.flatnonzero(kind==source)
            if len(indices):
                target=torch.as_tensor(indices,device='cuda',dtype=torch.long)
                selected=torch.as_tensor(slots[indices],device='cuda',dtype=torch.long)
                packed[target]=bank[selected]
        capture_root=getattr(m.runtime,'nw_capture_dir',None)
        if capture_root is not None and m.layer==0 and m.calls==0:
            from pathlib import Path
            path=Path(capture_root)
            path.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path/'layer0_source.npz',
                rms=n.rms.detach().cpu().numpy(),
                residual=self.final_residual.detach().cpu().numpy(),
                output=self.y.detach().cpu().numpy(),
                hidden=self.h[:count*n.group_size].detach().cpu().numpy(),
                descriptors=desc,packed=packed.cpu().numpy(),
                scales=m.reader.scales[physical])
        ref=gpu_dequantize(packed.reshape(count*n.group_size,3,self.width//2),
                           torch.from_numpy(m.reader.scales[physical]).cuda())
        gate=torch.nn.functional.linear(n.rms,ref[:,0]);up=torch.nn.functional.linear(n.rms,ref[:,1])
        hidden=torch.nn.functional.silu(gate)*up
        expected=hidden@ref[:,2]
        neuron_dest=np.stack((dest*2,dest*2+1),axis=1).ravel()
        actual_hidden=self.h[torch.as_tensor(neuron_dest,device='cuda',dtype=torch.long)]
        torch.testing.assert_close(actual_hidden,hidden.reshape(-1),rtol=.02,atol=.02)
        error=(self.y-expected).abs().float()
        scale=expected.abs().float()
        max_abs=float(error.max());mean_abs=float(error.mean())
        relative_rms=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(scale).clamp_min(1e-12))
        if max_abs>0.2 or relative_rms>0.01:
            raise AssertionError(f'No-waste packed output mismatch: max_abs={max_abs:.6g}, '
                                 f'mean_abs={mean_abs:.6g}, relative_rms={relative_rms:.6g}')
        actual=self.h.detach().abs().float().cpu().numpy().reshape(-1,2).sum(1)
        by_dest=np.empty(count,np.int32);by_dest[dest]=groups
        order=np.lexsort((by_dest,-actual[:count]))[:n.active]
        self.no_waste_expected_retained=by_dest[order].copy()
        self.no_waste_expected_keep=self.no_waste_expected_retained[:n.core].copy()
        self.no_waste_pre_cache_groups=n.view(m.cache,0,n.active).copy()
        self.no_waste_validations=getattr(self,'no_waste_validations',[])
        self.no_waste_validations.append(dict(layer=m.layer,exec_groups=count,
            optional_groups=optional,packed_fixed_exec='PASS',group_ids_unique='PASS',
            output_max_abs=max_abs,output_mean_abs=mean_abs,output_relative_rms=relative_rms))

    def validate_no_waste_retention(self,m):
        import numpy as np
        n=self.n
        actual_selected=n.view(m.cache,3,n.active).copy()
        actual_keep=n.view(m.cache,2,n.core).copy()
        np.testing.assert_array_equal(actual_selected,self.no_waste_expected_retained)
        np.testing.assert_array_equal(actual_keep,self.no_waste_expected_keep)
        if len(np.unique(n.view(m.cache,0,n.active)))!=n.active:
            raise AssertionError('Late cache slots are not unique')
        pre_resident=set(map(int,self.no_waste_pre_cache_groups))
        new_groups=[int(g) for g in self.no_waste_expected_retained if int(g) not in pre_resident]
        if new_groups:
            group_to_slot=n.view(m.cache,1,7168)
            slots=torch.as_tensor([int(group_to_slot[g]) for g in new_groups],device='cuda',dtype=torch.long)
            group_ids=torch.as_tensor(new_groups,device='cuda',dtype=torch.long)
            packed_cache=m.packed_cache.reshape(n.active,-1)
            packed_staging=self.staging.reshape(7168,-1)
            if not torch.equal(packed_cache[slots],packed_staging[group_ids]):
                raise AssertionError('Late cache copy differs from staged packed source')
        self.no_waste_validations[-1].update(late_retained='PASS',next_keep='PASS',
                                             staged_cache_copy='PASS',staged_cache_copy_groups=len(new_groups))

class Norm(MappedNorm):
    def forward(self,x):
        # Coarse dispatch only; the bank and engine are per layer.
        self.mlp.runtime.split=self.mlp.engine
        return super().forward(x)

class MLP(MappedMLP):
    def layer_begin(self,module,inputs):
        self.was_prefill=self.prefill
        if self.prefill:return
        n=self.native
        if self.runtime.diagnostic:self.events["layer"][0].record()
        n.check(n.lib.lp_maintenance_wait(n.ctx,1)) # Unexpected SAME-layer reuse only; measured/countable.
        # Diagnostic-only: P2 consumes this KEEP snapshot. Maintenance after
        # this call prepares a *different* KEEP set for the next token.
        if getattr(self.runtime,'semantic_mid_samples',None) is not None:
            self.semantic_keep_current=n.view(self.cache,2,n.core).copy()
        if getattr(self.runtime,'nw_native_set_audit',False):
            self.nw_current_keep=n.view(self.cache,2,n.core).copy()
        if self.runtime.p0_budget:
            n.check(n.lib.lp_p0_start(n.ctx,self.layer))
        self.runtime.native=n;self.runtime.split=self.engine
        self.row=dict(layer=self.layer,token=self.calls,_events={})
        # Retain the device source until this layer is reused. The maintenance
        # dependency above proves the previous P1/P2 consumers have finished.
        self.input_source=inputs[0]
        if not self.runtime.producer_publication or self.layer==0:
            self.engine.input_ready.record()
            with torch.cuda.stream(self.runtime.input_publication):
                self.runtime.input_publication.wait_event(self.engine.input_ready)
                n.check(n.lib.lm_hidden_nowait(n.mapped,self.input_source.data_ptr(),0,
                                            self.runtime.input_publication.cuda_stream))
                self.engine.input_done.record()
        n.check((n.lib.lp_control_arm if self.runtime.ready_scheduler else n.lib.lp_control_submit)(
            n.ctx,1,self.layer if self.runtime.ready_scheduler else 0))
        if self.runtime.publication_check:
            n.check(n.lib.lm_wait(n.mapped,0))
            torch.testing.assert_close(n.hidden[0].view(torch.int16),inputs[0].view(torch.int16),rtol=0,atol=0)
            self.runtime.publication_checks.append(dict(layer=self.layer,token=self.calls,group_size=n.group_size,status='BIT_EXACT'))
        if self.runtime.validate and not self.hidden_checked:
            n.check(n.lib.lm_wait(n.mapped,0))
            torch.testing.assert_close(n.hidden[0],inputs[0],rtol=0,atol=0);self.hidden_checked=True
        # HF consumes its original CUDA device tensor immediately. No mapped alias or CPU wait.
    def layer_end(self,module,inputs,output):
        if not self.was_prefill and self.runtime.diagnostic:self.events["layer"][1].record()
        return None # Preserve ordinary device residual and HF output structure.
    def attention_begin(self,module,args,kwargs):
        if not self.prefill and self.runtime.diagnostic:self.events['attention'][0].record()
        super().attention_begin(module,args,kwargs)
    def attention_end(self,module,args,kwargs,output):
        super().attention_end(module,args,kwargs,output)
        if not self.prefill and self.runtime.diagnostic:self.events['attention'][1].record()
    def seed_cache(self):
        super().seed_cache()
        self.native.check(self.native.lib.lp_prepare_a(self.native.ctx,self.layer))

class Runtime(BaseRuntime):
    def __init__(self,model,root,predictor_library,library,timing=False,validate=False,split=True,final_qd=2,speculate=False,legacy_split=False,diagnostic=False,layer_configs=None,group_root=None,model_spec=None,predictor_root=None,predictor_dim=1024,spec_p1_size=0,spec_mid_size=0,mid_partial_root=None,ready_scheduler=False,ready_notification='poll',p0_budget=0,p1_lite_root=None,p1_lite_budget=0,p1_lite_features=0,tiny_mid_features=0,semantic_mid_rescue=0,relaxed_p2_m=0,relaxed_p2_p=0,no_waste_policy='strict'):
        if not split:raise ValueError('Use the unchanged base runtime for split=False')
        if timing:raise ValueError('This integration uses application-level timing')
        self.publication_method="fused";self.producer_publication=False;self.publication_check=False;self.publication_checks=[]
        self.model_thread_affinity=os.sched_getaffinity(0)
        self.model=model;self.root=Path(root);self.timing=False;self.validate=validate;self.diagnostic=False
        self.predictor_library=predictor_library;self.records=[];self.layers=[]
        self.model_spec=model_spec;self.predictor_root=Path(predictor_root) if predictor_root else None;self.predictor_dim=predictor_dim
        self.spec_p1_size=spec_p1_size;self.spec_mid_size=spec_mid_size
        self.p0_budget=p0_budget;self.p1_lite_root=Path(p1_lite_root) if p1_lite_root else None
        self.p1_lite_budget=p1_lite_budget;self.p1_lite_features=p1_lite_features
        self.tiny_mid_features=tiny_mid_features
        self.semantic_mid_rescue=semantic_mid_rescue
        self.relaxed_p2_m=relaxed_p2_m;self.relaxed_p2_p=relaxed_p2_p
        if no_waste_policy not in ('strict','knee','conservative','strictreads',
                                   'diag_m1','diag_m2','diag_m3'):
            raise ValueError('Unsupported no-waste policy')
        self.no_waste_policy=no_waste_policy
        if no_waste_policy!='strict' and (p0_budget!=32 or p1_lite_budget!=128 or
                spec_p1_size!=1024 or p1_lite_features!=512 or spec_mid_size or
                tiny_mid_features or semantic_mid_rescue or relaxed_p2_m or relaxed_p2_p):
            raise ValueError('No-waste requires accepted P0-32/P1-lite A-128 and no Mid/relaxed P2')
        if (relaxed_p2_m,relaxed_p2_p) not in ((0,0),(247,192),(279,160)):
            raise ValueError('Unsupported bounded relaxed P2 policy')
        if semantic_mid_rescue and (semantic_mid_rescue!=16 or p0_budget!=32 or p1_lite_budget!=128 or spec_mid_size or tiny_mid_features):
            raise ValueError('Semantic Mid rescue requires P0-32, lite-128, and no other Mid')
        if (p0_budget or p1_lite_budget) and (not ready_scheduler or ready_notification!='doorbell' or
                (spec_mid_size and not tiny_mid_features)):
            raise ValueError('Early speculation requires doorbell READY scheduler and no Mid')
        if tiny_mid_features and (tiny_mid_features not in (64,128) or not p0_budget or
                p1_lite_budget!=128 or spec_mid_size!=128 or mid_partial_root is None):
            raise ValueError('Tiny residual Mid requires P0, P1-lite-128, Mid-128 artifacts')
        if p0_budget and not spec_p1_size:
            raise ValueError('P0 requires the bound model-specific speculative hot order')
        if p1_lite_budget and p1_lite_root is None:
            raise ValueError('Compact P1 requires model-specific packed artifacts')
        self.mid_partial_root=Path(mid_partial_root) if mid_partial_root else None
        self.ready_scheduler=ready_scheduler
        if ready_notification not in ('poll','doorbell') or (ready_notification!='poll' and not ready_scheduler):
            raise ValueError('Doorbell notification requires the ready scheduler')
        self.ready_notification=ready_notification
        decoder=model.model.layers
        if model_spec is not None:
            if len(decoder)!=model_spec['layers'] or model.config.hidden_size!=model_spec['H'] or model.config.intermediate_size!=model_spec['I']:
                raise ValueError('Model geometry does not match runtime artifact')
            if self.predictor_root is None:raise ValueError('New-model predictor root required')
        self.maintenance=torch.cuda.Stream(priority=0)
        self.input_publication=torch.cuda.Stream(priority=0)
        configs=layer_configs or [dict(group_size=2,stable=False) for _ in decoder]
        if len(configs)!=len(decoder):raise ValueError("Layer-policy count mismatch")
        parent=None
        for i,layer in enumerate(decoder):
            if type(layer.self_attn).__name__ not in ('MistralSdpaAttention','LlamaSdpaAttention'):raise ValueError('HF SDPA required')
            cfg=configs[i]
            self.native=Native(library,predictor_library,parent,group_size=cfg["group_size"],stable=cfg.get("stable",True),model_spec=model_spec,predictor_stride=predictor_dim)
            if relaxed_p2_m:
                lib=self.native.lib
                lib.lp_relaxed_p2_config.argtypes=[C.c_void_p,C.c_int,C.c_int]
                lib.lp_relaxed_p2_config.restype=C.c_int
                self.native.check(lib.lp_relaxed_p2_config(self.native.ctx,relaxed_p2_m,relaxed_p2_p))
            if spec_p1_size:
                lib=self.native.lib
                lib.lp_speculative_config.argtypes=[C.c_void_p,C.c_int,C.c_int]
                lib.lp_speculative_config.restype=C.c_int
                self.native.check(lib.lp_speculative_config(self.native.ctx,spec_p1_size,spec_mid_size))
            if p0_budget or p1_lite_budget:
                lib=self.native.lib
                lib.lp_early_speculation_config.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int]
                lib.lp_early_speculation_config.restype=C.c_int
                self.native.check(lib.lp_early_speculation_config(self.native.ctx,p0_budget,p1_lite_budget,p1_lite_features))
            if no_waste_policy!='strict':
                lib=self.native.lib
                lib.lp_no_waste_config.argtypes=[C.c_void_p,C.c_int]
                lib.lp_no_waste_config.restype=C.c_int
                self.native.check(lib.lp_no_waste_config(self.native.ctx,
                    {'knee':1,'conservative':2,'strictreads':3,
                     'diag_m1':4,'diag_m2':5,'diag_m3':6}[no_waste_policy]))
            if tiny_mid_features:
                lib=self.native.lib
                lib.lp_tiny_mid_config.argtypes=[C.c_void_p,C.c_int]
                lib.lp_tiny_mid_config.restype=C.c_int
                self.native.check(lib.lp_tiny_mid_config(self.native.ctx,tiny_mid_features))
            if semantic_mid_rescue:
                lib=self.native.lib
                lib.lp_semantic_mid_config.argtypes=[C.c_void_p,C.c_int]
                lib.lp_semantic_mid_config.restype=C.c_int
                self.native.check(lib.lp_semantic_mid_config(self.native.ctx,semantic_mid_rescue))
            if parent is None:parent=self.native
            self.split=Engine(self.native,no_waste=no_waste_policy!='strict')
            if group_root is None:m=MLP(self,layer.mlp,i)
            else:
                raise NotImplementedError("Optional model adapter supports uniform Group2 layouts")
                m=MixedMLP(self,layer.mlp,i,Path(group_root),cfg)
            m.engine=self.split;self.layers.append(m)
            layer.mlp=m;layer.post_attention_layernorm=Norm(layer.post_attention_layernorm,m)
            layer.register_forward_pre_hook(m.layer_begin);layer.register_forward_hook(m.layer_end)
            layer.self_attn.register_forward_pre_hook(m.attention_begin,with_kwargs=True)
            layer.self_attn.register_forward_hook(m.attention_end,with_kwargs=True)
        for i in range(len(self.layers)-1):
            engine=self.layers[i].engine;target=self.layers[i+1].native
            engine.next_native=target;engine.next_hidden=target.hidden[0]
            engine.next_flag=capsule(target.lib.lm_hidden_flag_dlpack(target.mapped))
        self.reset()
        warmed=set()
        for m in self.layers:
            if m.native.group_size not in warmed:m.engine.warm(m);warmed.add(m.native.group_size)
        for m in self.layers:
            m.engine.warmed=True
            m.native.check(m.native.lib.lp_speculation(m.native.ctx,int(speculate)))
            m.native.check(m.native.lib.lp_final_qd(m.native.ctx,final_qd))
            m.native.notify_bell=None
            if ready_scheduler:
                if ready_notification=='doorbell':
                    m.native.check(m.native.lib.lp_ready_notification_mode(m.native.ctx,1))
                m.native.check(m.native.lib.lp_ready_scheduler_enable(m.native.ctx))
                if ready_notification=='doorbell':
                    m.native.lib.lm_notify_doorbell_dlpack.argtypes=[C.c_void_p]
                    m.native.lib.lm_notify_doorbell_dlpack.restype=C.c_void_p
                    m.native.notify_bell=capsule(m.native.lib.lm_notify_doorbell_dlpack(m.native.mapped))
            m.native.check(m.native.lib.lp_async_enable(m.native.ctx))
        if diagnostic:
            for m in self.layers:
                m.native.lib.lp_diagnostic_trace.argtypes=[C.c_void_p,C.c_int]
                m.native.lib.lp_diagnostic_trace.restype=None
                m.native.lib.lp_control_trace.argtypes=[C.c_void_p]
                m.native.lib.lp_control_trace.restype=C.POINTER(C.c_uint64)
                m.native.lib.lp_diagnostic_trace(m.native.ctx,1)
            self.token_end=torch.cuda.Event(enable_timing=True);self.token_end.record()
            for m in self.layers:
                m.events={k:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                          for k in ('layer','attention','norm','A','B_including_wait','reduction',
                                    'publication','packed_cache_update','mapped_importance')}
                for pair in m.events.values():
                    for event in pair:event.record() # Initialize handles outside all measurements.
                a,b=m.events['norm'];m.native.lib.si_profile_events(m.native.shared,a.cuda_event,b.cuda_event)
            self.diagnostic=True
    @contextmanager
    def event(self,row,key):
        if not self.diagnostic:yield;return
        events=self.layers[row['layer']].events[key]
        events[0].record();yield;events[1].record()
    def diagnostic_rows(self):
        if not self.diagnostic:return []
        rows=[]
        for i,m in enumerate(self.layers):
            e=m.events
            row={k:a.elapsed_time(b) for k,(a,b) in e.items() if k!="layer"}
            row.update(layer=i,layer_total_gpu_ms=e['layer'][0].elapsed_time(e['layer'][1]),
                       pre_attention_gpu_ms=e['layer'][0].elapsed_time(e['attention'][0]),
                       attention_to_norm_gpu_ms=e['attention'][1].elapsed_time(e['norm'][0]),
                       post_reduction_gpu_ms=e['reduction'][1].elapsed_time(e['layer'][1]),norm_to_A=e['norm'][1].elapsed_time(e['A'][0]),
                       A_to_B=e['A'][1].elapsed_time(e['B_including_wait'][0]),
                       B_to_reduction=e['B_including_wait'][1].elapsed_time(e['reduction'][0]))
            if i<len(self.layers)-1:row['next_layer_gap']=e['layer'][1].elapsed_time(self.layers[i+1].events['layer'][0])
            stamp=m.engine.stamp.host.reshape(m.engine.programs,4);pa=m.engine.pa
            row['B_exposed_wait']=max(0,int(stamp[pa:,1].min())-int(stamp[pa:,0].min()))/1e6
            row['B_compute_timestamp_ms']=(int(stamp[pa:,3].max())-int(stamp[pa:,2].min()))/1e6
            row['A_to_B_dispatch_gap_ms']=max(0,int(stamp[pa:,0].min())-int(stamp[:pa,3].max()))/1e6
            # Event-bracketed B includes dispatch and completion publication as well as math.
            row['B_nonpoll_event_ms']=row['B_including_wait']-row['B_exposed_wait']
            row['native_times']=m.native.times.copy().tolist()
            row['control_ns']=[int(m.native.lib.lp_control_trace(m.native.ctx)[j]) for j in range(32)]
            rows.append(row)
        rows[-1]['model_head_gpu_ms']=self.layers[-1].events['layer'][1].elapsed_time(self.token_end)
        rows[-1]['gpu_decode_span_ms']=self.layers[0].events['layer'][0].elapsed_time(self.token_end)
        return rows
    def drain(self):
        for m in self.layers:
            m.native.check(m.native.lib.lp_control_wait(m.native.ctx))
            m.native.check(m.native.lib.lp_maintenance_wait(m.native.ctx,0))
    def reset(self):
        self.drain()
        for m in self.layers:
            m.native.check(m.native.lib.lp_reset(m.native.ctx));m.reset()
        # ls_create pins its caller as a predictor master. Prediction now runs
        # on ControlQueue's independently pinned 12-core team, not this thread.
        # Also undo pins left by validation-only reference predictors.
        os.sched_setaffinity(0,self.model_thread_affinity)
    def counters(self):
        return dict(main_waits=sum(m.native.lib.lp_main_waits(m.native.ctx) for m in self.layers),
                    main_wait_ms=sum(m.native.lib.lp_main_wait_ms(m.native.ctx) for m in self.layers))
