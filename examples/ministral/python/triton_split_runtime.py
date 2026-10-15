"""Legacy/reference 85/15 path and shared validated kernel helpers.
The production asynchronous path is python.split_async. No changes to attention or selection.
One persistent mapped descriptor/epoch arena; one native final-I/O publication
worker. CUDA stream order: RMS -> A -> B(acquire) -> packed cache -> reduction
-> importance. Ownership ends only after importance acquire on the CPU.
"""
import ctypes as C
import os
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import triton
from python.two_stage_mapped import capsule
from python.triton_runtime_kernels import runtime_split
from python.triton_split_mlp import reduce_split
from scripts.triton_split_probe import Mapped,library

TIME_KEYS=('p1_quant','p1_w4a8','p1_total','p1_publish','p2_quant','p2_w4a8','p2_total','p2_publish',
 'native_plan','mlp_io_wait','native_keep','p1_requested','loaded_before_p2','ready_at_p2','final_missing',
 'fill_overlap','speculative_waste','pread_calls','rms_ready_wait','p2_update_active','p1_cache_prefetch',
 'p1_pio_stats','p1_pio_update','p1_wait_started')

class SplitEngine:
    def __init__(self,n,no_waste=False):
        self.g=n.group_size;self.core=n.core;self.fill=n.fill;self.active=n.active
        self.public_mode=os.environ.get('LEANSTREAM_PUBLIC_CONSUMER','progressive')
        self.no_waste=bool(no_waste)
        if self.no_waste and (self.g,self.core,self.fill,self.active)!=(2,1219,215,1434):
            raise ValueError('This backend requires validated Ministral Group2 geometry')
        self.width=int(n.hidden[0].shape[-1])
        self.warps=int(os.environ.get('LEANSTREAM_SPLIT_WARPS','4'))
        if self.warps not in (4,8):raise ValueError('Unsupported split warp count')
        self.ready_ids=int(os.environ.get('LEANSTREAM_READY_IDS','0'))
        self.execution=os.environ.get('LEANSTREAM_EXEC_QUEUE','0')=='1' or self.ready_ids!=0
        if self.execution and type(self) is SplitEngine:
            raise ValueError('Execution queue requires the asynchronous stream_io_v2 runtime')
        if self.no_waste and self.execution:raise ValueError('No-waste does not support the execution-queue variant')
        self.max_optional=self.fill+32+128+self.fill if self.no_waste else self.fill
        self.chunk=self.max_optional if self.no_waste else (self.fill if self.ready_ids else (int(os.environ.get('LEANSTREAM_EXEC_CHUNK','16')) if self.execution else self.fill))
        if not 1<=self.chunk<=self.max_optional:raise ValueError('Invalid dynamic execution chunk')
        self.chunk_groups=[min(self.chunk,self.max_optional-i) for i in range(0,self.max_optional,self.chunk)]
        self.pa=(self.core*self.g+3)//4;self.pb=(self.max_optional*self.g+3)//4;self.programs=self.pa+self.pb
        if self.public_mode=='progressive':self.pb=self.max_optional;self.programs=self.pa+self.pb
        if self.execution:
            self.pb=sum((count*self.g+3)//4 for count in self.chunk_groups);self.programs=self.pa+self.pb
        self.n=n;l=n.lib;p=C.c_void_p;i=C.c_int
        signatures={'lp_split_bind':([p,p,p,p],i),'lp_split_start':([p,p,p,C.c_float,p],C.c_uint32),
                    'lp_split_p2':([p],i),'lp_split_update':([p,p,p],i),'lp_split_keep':([p],i),
                    'lm_packed_dlpack':([p],p),'lp_ready_times':([p],C.POINTER(C.c_uint64))}
        for name,(args,result) in signatures.items():f=getattr(l,name);f.argtypes=args;f.restype=result
        lib=library(None)
        if self.execution:
            self.queue=Mapped(lib,self.active*5,np.int32)
            self.a=SimpleNamespace(host=self.queue.host[:self.core*5],tensor=self.queue.tensor[:self.core*5])
            self.b=SimpleNamespace(host=self.queue.host[self.core*5:],tensor=self.queue.tensor[self.core*5:])
        else:self.a=Mapped(lib,self.core*5,np.int32);self.b=Mapped(lib,self.max_optional*5,np.int32)
        self.ready=Mapped(lib,3 if self.no_waste else len(self.chunk_groups),np.int32);self.started=Mapped(lib,self.programs,np.int32)
        self.done=Mapped(lib,self.programs,np.int32);self.stamp=Mapped(lib,self.programs*4,np.int64)
        self.reduced=Mapped(lib,self.width//32,np.int32)
        self.staging=capsule(l.lm_packed_dlpack(n.mapped))
        self.h=torch.empty((self.active+32+128+self.fill if self.no_waste else self.active)*self.g,dtype=torch.float16,device='cuda')
        self.partial=torch.empty((self.programs,self.width),dtype=torch.float32,device='cuda')
        self.y=torch.empty((1,1,self.width),dtype=torch.float16,device='cuda')
        if self.execution:
            l.lp_exec_bind.argtypes=[p,p,p,i,i];l.lp_exec_bind.restype=i
            n.check(l.lp_exec_bind(n.ctx,self.queue.host.ctypes.data,self.ready.host.ctypes.data,self.chunk,int(os.environ.get('LEANSTREAM_EXEC_PROFILE','0')=='1')))
            if self.ready_ids==3:
                self.execution_tail=Mapped(lib,1,np.int64)
                l.lp_ready_id_tail.argtypes=[p,p];l.lp_ready_id_tail.restype=i
                n.check(l.lp_ready_id_tail(n.ctx,self.execution_tail.host.ctypes.data))
        else:n.check(l.lp_split_bind(n.ctx,self.a.host.ctypes.data,self.b.host.ctypes.data,self.ready.host.ctypes.data))
        if self.no_waste:
            l.lp_no_waste_count_bind.argtypes=[p,p]
            l.lp_no_waste_count_bind.restype=i
            n.check(l.lp_no_waste_count_bind(n.ctx,self.ready.host.ctypes.data+4))
        self.dynamic=[];offset=self.pa;first=0
        for k,count in enumerate(self.chunk_groups):
            grid=(count*self.g+3)//4
            self.dynamic.append((self.b.tensor[first*5:(first+count)*5],self.ready.tensor[k:k+1],count*self.g,offset,grid))
            offset+=grid;first+=count
        self.public_mode=os.environ.get('LEANSTREAM_PUBLIC_CONSUMER','progressive')
        if self.public_mode not in ('old','batch','progressive'):raise ValueError('Invalid consumer mode')
        if self.public_mode!='old':
            if not self.no_waste:raise ValueError('Production public probe uses current KNEE geometry')
            self.public_desc=Mapped(lib,self.max_optional*5,np.int32) if self.public_mode=='progressive' else self.b
            self.public_indices=Mapped(lib,self.max_optional,np.int32)
            self.public_control=Mapped(lib,32,np.int32)
            self.public_claim=torch.zeros(3,device="cuda",dtype=torch.int32)
            self.public_exec_counts=Mapped(lib,self.max_optional,np.int32) if os.environ.get('LEANSTREAM_PUBLIC_EXECUTION_CHECK')=='1' else None
            self.public_launch_epoch=0
            self.workers=int(os.environ.get("LEANSTREAM_PUBLIC_WORKERS","32"))
            if not 1<=self.workers<=self.pb:raise ValueError("Invalid persistent worker count")
            l.lp_public_bind.argtypes=[p,p,p,p,i,i];l.lp_public_bind.restype=i
            n.check(l.lp_public_bind(n.ctx,self.public_desc.host.ctypes.data,self.public_indices.host.ctypes.data,self.public_control.host.ctypes.data,self.max_optional,int(self.public_mode=='progressive')))
        self.epoch=0;self.validations=[];self.warmed=False

    def launch(self,m,epoch):
        self.launch_keep(m,epoch);self.launch_dynamic(m,epoch)

    def launch_keep(self,m,epoch):
        common=(self.n.rms,m.packed_cache,self.staging,m.reader.device_scales)
        with m.runtime.event(m.row,'A'):
            self.public_a_kernel=runtime_split[(self.pa,)](*common,self.a.tensor,self.h,self.partial,self.ready.tensor,self.started.tensor,
                self.done.tensor,self.stamp.tensor,epoch,self.core*self.g,0,False,self.g,self.width,num_warps=self.warps,num_stages=1)

    def launch_dynamic(self,m,epoch):
        common=(self.n.rms,m.packed_cache,self.staging,m.reader.device_scales)
        if self.public_mode=='progressive':
            from backends.triton.persistent import persistent_split
            verify_args={}
            if True and self.public_exec_counts is not None:
                self.public_exec_counts.host.fill(0);verify_args={'Counts':self.public_exec_counts.tensor,'CHECK':True}
            self.public_launch_epoch+=1
            if self.public_launch_epoch>=2**31:raise RuntimeError("Queue launch epoch exhausted")
            with m.runtime.event(m.row,'B_including_wait'):
                self.public_b_kernel=persistent_split[(self.workers,)](*common,self.public_desc.tensor,self.h,self.partial,
                    self.public_indices.tensor,self.public_control.tensor,self.public_claim,self.done.tensor,self.stamp.tensor,epoch,self.public_launch_epoch,
                    self.pa,self.pb,self.workers,self.g,self.width,num_warps=self.warps,num_stages=1,**verify_args)
            return
        kernel=runtime_split
        if self.public_mode=='batch':
            raise ValueError("Optional model adapter requires Progressive mode")
        with m.runtime.event(m.row,'B_including_wait'):
            for desc,flag,count,offset,grid in self.dynamic:
                self.last_kernel=kernel[(grid,)](*common,desc,self.h,self.partial,flag,self.started.tensor,
                    self.done.tensor,self.stamp.tensor,epoch,count,offset,True,self.g,self.width,
                    self.ready.tensor[1:2] if self.no_waste else None,self.no_waste,
                    num_warps=self.warps,num_stages=1)

        self.public_b_kernel=self.last_kernel
    def release_chunks(self,epoch):
        for k in range(len(self.chunk_groups)):self.ready.lib.tm_release(self.ready.context,k,epoch)

    def warm(self,m):
        # Only initialization: valid repeated slot-zero descriptors, independent of cache contents.
        self.a.host.fill(0);self.b.host.fill(0)
        self.a.host.reshape(-1,5)[:,3]=np.arange(self.core)
        self.b.host.reshape(-1,5)[:,3]=0 if self.no_waste else np.arange(self.core,self.active)
        if self.no_waste:self.ready.host[1:3]=0
        timing=m.runtime.timing;m.runtime.timing=False;m.row={}
        self.release_chunks(1)
        if self.public_mode!='old':
            self.public_control.host.fill(0);self.public_control.host[3]=1
        self.launch(m,1)
        reduce_split[(self.width//32,)](self.partial,self.y,self.reduced.tensor,1,self.programs,
            triton.next_power_of_2(self.programs),32,WIDTH=self.width,num_warps=4,num_stages=1)
        self.reduced.wait(1);self.release_chunks(0);self.warmed=True;m.runtime.timing=timing

    def enqueue(self,m,residual,norm):
        if not self.warmed:raise RuntimeError('Triton must be warmed before decode')
        n=self.n
        self.epoch=n.lib.lp_split_start(n.ctx,residual.data_ptr(),norm.weight.data_ptr(),norm.variance_epsilon,
                                     torch.cuda.current_stream().cuda_stream)
        if not self.epoch:raise RuntimeError(n.lib.lp_error().decode())
        # Norm is already enqueued on THIS stream. A/B cannot precede the actual RMS producer.
        # B's epoch is still invalid: the P2 prediction has not even run yet.
        assert self.ready.host[0]!=self.epoch
        checking=m.runtime.validate and m.layer in (0,15,31) and not getattr(m,'split_checked',False)
        if checking:
            before=n.view(m.cache,0,self.active).copy();keep=n.view(m.cache,2,self.core).copy()
        self.launch(m,self.epoch)
        n.check(n.lib.lp_split_p2(n.ctx)) # Predict/plan/publish only; final I/O is native asynchronous work.
        if checking:
            fill=np.ctypeslib.as_array((C.c_int32*self.fill).from_address(n.lib.lp_fill(n.ctx,1))).copy()
            selected=np.concatenate((keep,fill));wanted=set(selected.tolist());cached=set(before.tolist())
            missing=np.array([g for g in selected if g not in cached],np.int32)
            slots=np.array([slot for slot,g in enumerate(before) if g not in wanted],np.int32)
            np.testing.assert_array_equal(n.view(m.cache,4,len(missing)),missing)
            np.testing.assert_array_equal(n.view(m.cache,5,len(slots)),slots)
            np.testing.assert_array_equal(n.view(m.cache,0,1434),before) # Still PLAN, not committed.
            np.testing.assert_array_equal(self.a.host.reshape(-1,5)[:,0],keep)


    def finish(self,m):
        n=self.n;stream=torch.cuda.current_stream().cuda_stream
        with m.runtime.event(m.row,'packed_cache_update'):
            n.check(n.lib.lp_split_update(n.ctx,m.packed_cache.data_ptr(),stream))
        with m.runtime.event(m.row,'reduction'):
            reduce_split[(128,)](self.partial,self.y,self.reduced.tensor,self.epoch,self.programs,1024,32,num_warps=4,num_stages=1)
        with m.runtime.event(m.row,'mapped_importance'):n.check(n.lib.lm_importance(n.mapped,self.h.data_ptr(),stream))
        n.check(n.lib.lp_split_keep(n.ctx))
        if m.runtime.timing:
            m.row.update(zip(TIME_KEYS,map(float,n.times)))
            stamps=self.stamp.host.reshape(self.programs,4)
            # GPU %globaltimer domain throughout. Only the wait AFTER useful A completion.
            a_end=int(stamps[:self.pa,3].max());b_first=int(stamps[self.pa:,0].min())
            b_acquire=int(stamps[self.pa:,1].min());b_end=int(stamps[self.pa:,3].max())
            cpu=np.ctypeslib.as_array(n.lib.lp_ready_times(n.ctx),shape=(6,))
            m.row.update(dict(zip(('last_required_pread_complete_cpu_ns','final_worker_wake_cpu_ns',
                'descriptor_ready_cpu_ns','ready_publish_before_cpu_ns','ready_publish_after_cpu_ns',
                'immutable_metadata_ready_cpu_ns'),map(int,cpu))))
            m.row.update(A_first_start_gpu_ns=int(stamps[:self.pa,0].min()),A_last_math_end_gpu_ns=a_end,
                B_first_entry_gpu_ns=b_first,B_first_successful_acquire_gpu_ns=b_acquire,
                B_compute_start_gpu_ns=int(stamps[self.pa:,2].min()),
                last_io_to_ready_ms=(int(cpu[3])-int(cpu[0]))/1e6 if cpu[0] else 0,
                publish_store_bracket_ns=int(cpu[4])-int(cpu[3]))
            m.row.update(B_poll_exposed_wait=max(0,b_acquire-b_first)/1e6,
                         B_compute=(b_end-b_acquire)/1e6,
                         A_to_B_dispatch_gap=max(0,b_first-a_end)/1e6,
                         early_B_enqueue=True)
        if m.runtime.validate and m.layer in (0,15,31) and not getattr(m,'split_checked',False):
            self.validate(m);m.split_checked=True
        return self.y

    def validate(self,m):
        # OFFLINE gate only, after GPU consumption; never in timing.
        from python.v1_storage import gpu_dequantize
        n=self.n;ids=n.view(m.cache,0,self.active).copy()
        desc=np.concatenate((self.a.host.reshape(-1,5),self.b.host.reshape(-1,5)))
        assert len(np.unique(desc[:,0]))==self.active
        assert len(np.unique(desc[:,3]))==self.active
        np.testing.assert_array_equal(ids[desc[:,3]],desc[:,0])
        physical=m.reader.mapping[m.groups[ids].ravel()]
        ref=gpu_dequantize(m.packed_cache.reshape(2868,3,2048),torch.from_numpy(m.reader.scales[physical]).cuda())
        gate=torch.nn.functional.linear(n.rms,ref[:,0]);up=torch.nn.functional.linear(n.rms,ref[:,1])
        h=torch.nn.functional.silu(gate)*up;y=h@ref[:,2]
        torch.testing.assert_close(self.y,y,rtol=.02,atol=.02)
        torch.testing.assert_close(self.h,h.reshape(-1),rtol=.02,atol=.02)
        error=float((self.y-y).abs().max())
        # Real SSD bytes, sampled physical cache slots; also covers B cache-hit descriptors.
        for slot in (0,self.active//2,self.active-1):
            packed,_,_=m.reader.read_packed(m.groups[ids[slot]])
            torch.testing.assert_close(m.packed_cache.reshape(self.active,-1)[slot].cpu(),packed.reshape(-1),rtol=0,atol=0)
        assert np.all(self.done.host==self.epoch) and np.all(self.reduced.host==self.epoch)
        # Replay identical final weights with ALL_READY, after the real delayed I/O path.
        # No storage requests; all final groups now reside in the GPU packed cache.
        saved=self.y.clone();saved_b=self.b.host.copy();saved_epoch=self.epoch
        bd=self.b.host.reshape(-1,5);bd[:,2]=bd[:,3];bd[:,4]=0
        replay=saved_epoch+1000000;self.release_chunks(replay)
        timing=m.runtime.timing;m.runtime.timing=False
        self.launch(m,replay)
        reduce_split[(128,)](self.partial,self.y,self.reduced.tensor,replay,self.programs,1024,32,num_warps=4,num_stages=1)
        self.reduced.wait(replay)
        torch.testing.assert_close(self.y,saved,rtol=0,atol=0)
        self.b.host[:]=saved_b;self.release_chunks(saved_epoch);m.runtime.timing=timing
        self.validations.append(dict(delayed_vs_all_ready='PASS', layer=m.layer,status='PASS',max_abs_error=error,groups_exact=True,packed_bytes_exact=True))
