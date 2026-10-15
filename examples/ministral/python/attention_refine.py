"""Unsplit direct-GQA attention and optional checkpoint configuration."""
from types import MethodType
import torch
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

class DirectGQAAttention:
    """Opt-in one-launch direct-GQA decode, preserving HF Q/K/V/RoPE/KV/o_proj."""
    def __init__(self,model,block=128,warps=4,capture_output=False,
                 checkpoint=False,runtime=None,midpoint=False,head_split16=False,
                 semantic_projection=False,projection_overlap=False,semantic_mid_publish=False,
                 semantic_capture_prefix=False):
        if block not in (64,128,256) or warps not in (4,8):
            raise ValueError('Unsupported bounded direct-GQA configuration')
        if checkpoint and runtime is None:raise ValueError('Checkpoint requires mapped runtime')
        if midpoint and not checkpoint:raise ValueError('Midpoint requires checkpoint')
        if head_split16 and checkpoint:raise ValueError('Head split gate is separate from KV-prefix checkpoint')
        if semantic_projection and not head_split16:raise ValueError('Semantic partial projection requires 16+16 heads')
        if projection_overlap and not semantic_projection:raise ValueError('Projection overlap requires semantic partial projection')
        if semantic_mid_publish and not semantic_projection:raise ValueError('Semantic Mid needs partial projection')
        if getattr(runtime,'semantic_mid_rescue',0) and (not semantic_mid_publish or projection_overlap):
            raise ValueError('Real semantic Mid requires serialized split and mapped publication')
        if semantic_capture_prefix and not semantic_mid_publish:raise ValueError('Prefix ablation needs semantic Mid capture')
        mid_stream=torch.cuda.Stream() if projection_overlap else None
        self.entries=[]
        for layer in model.model.layers:
            a=layer.self_attn
            if (a.num_heads,a.num_key_value_heads,a.head_dim)!=(32,8,128):
                raise ValueError('Unexpected Ministral attention geometry')
            entry=dict(original=a.forward,block=block,warps=warps,midpoint=midpoint,
                       head_split16=head_split16,diagnostic=bool(runtime and runtime.diagnostic),
                       phase_events=[],semantic_projection=semantic_projection,
                       projection_overlap=projection_overlap,mid_stream=mid_stream,
                       semantic_mid_publish=semantic_mid_publish,
                       semantic_mid_io=bool(getattr(runtime,'semantic_mid_rescue',0)),
                       ready_scheduler=bool(getattr(runtime,'ready_scheduler',False)),
                       capture_output=capture_output,last_output=None,
                       z=torch.empty((1,1,4096),device='cuda',dtype=torch.float16))
            if semantic_projection:
                if a.o_proj.bias is not None:raise ValueError('Semantic projection requires bias-free o_proj')
                entry.update(weight=a.o_proj.weight,pa=torch.empty((4096,),device='cuda',dtype=torch.float32),
                             pb=torch.empty((4096,),device='cuda',dtype=torch.float32),
                             y=torch.empty((1,1,4096),device='cuda',dtype=torch.float16))
            if semantic_mid_publish:
                import ctypes as C
                n=runtime.layers[a.layer_idx].native
                n.lib.lm_mid_norm.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_float,C.c_void_p]
                n.lib.lm_mid_norm.restype=C.c_int
                entry.update(native=n,mlp=runtime.layers[a.layer_idx])
            if semantic_capture_prefix:
                entry['prefix_z']=torch.empty((1,1,4096),device='cuda',dtype=torch.float16)
            if checkpoint:
                import ctypes as C
                from python.two_stage_mapped import capsule
                n=runtime.layers[a.layer_idx].native
                n.lib.lm_mid_checkpoint_reserve.argtypes=[C.c_void_p]
                n.lib.lm_mid_checkpoint_reserve.restype=C.c_uint32
                n.lib.lm_mid_checkpoint_flags_dlpack.argtypes=[C.c_void_p]
                n.lib.lm_mid_checkpoint_flags_dlpack.restype=C.c_void_p
                entry.update(native=n,mid=n.hidden[1],
                             meta=capsule(n.lib.lm_mid_checkpoint_flags_dlpack(n.mapped)),
                             bell=getattr(n,'notify_bell',None))
            a._direct_gqa=entry;a.forward=MethodType(direct_gqa_forward,a)
            self.entries.append(entry)

    def rows(self):
        return [({name:start.elapsed_time(end) for name,(start,end) in e['phase_events'].items()}
                 if isinstance(e['phase_events'],dict) else
                 [start.elapsed_time(end) for start,end in e['phase_events']])
                for e in self.entries]
    def outputs(self):
        return [entry['last_output'].detach().cpu().numpy().copy() for entry in self.entries]

def direct_gqa_forward(self,hidden_states,attention_mask=None,position_ids=None,
                       past_key_value=None,output_attentions=False,use_cache=False,**kwargs):
    entry=self._direct_gqa
    from transformers.cache_utils import DynamicCache
    if (hidden_states.shape[:2]!=(1,1) or hidden_states.dtype!=torch.float16 or
            hidden_states.device.type!='cuda' or self.training or attention_mask is not None or
            output_attentions or not isinstance(past_key_value,DynamicCache)):
        result=entry['original'](hidden_states,attention_mask,position_ids,past_key_value,
                                 output_attentions,use_cache,**kwargs)
        if entry['capture_output'] and hidden_states.shape[:2]==(1,1):entry['last_output']=result[0]
        return result
    from python.attention_decode_kernels import gqa_decode
    if entry['diagnostic']:
        prefix_events={}
        def prefix_timed(name,fn):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            value=fn()
            end.record();prefix_events[name]=(start,end)
            return value
        q=prefix_timed('q_proj',lambda:self.q_proj(hidden_states)).view(1,1,32,128).transpose(1,2)
        k=prefix_timed('k_proj',lambda:self.k_proj(hidden_states)).view(1,1,8,128).transpose(1,2)
        v=prefix_timed('v_proj',lambda:self.v_proj(hidden_states)).view(1,1,8,128).transpose(1,2)
    else:
        q=self.q_proj(hidden_states).view(1,1,32,128).transpose(1,2)
        k=self.k_proj(hidden_states).view(1,1,8,128).transpose(1,2)
        v=self.v_proj(hidden_states).view(1,1,8,128).transpose(1,2)
    length=k.shape[-2]+past_key_value.get_usable_length(k.shape[-2],self.layer_idx)
    if entry['diagnostic']:
        entry['last_kv_length']=length
        cos,sin=prefix_timed('rotary_emb',lambda:self.rotary_emb(v,seq_len=length))
        q,k=prefix_timed('rope',lambda:apply_rotary_pos_emb(q,k,cos,sin,position_ids))
        k,v=prefix_timed('kv_update',lambda:past_key_value.update(k,v,self.layer_idx,{'sin':sin,'cos':cos}))
    else:
        cos,sin=self.rotary_emb(v,seq_len=length)
        q,k=apply_rotary_pos_emb(q,k,cos,sin,position_ids)
        k,v=past_key_value.update(k,v,self.layer_idx,{'sin':sin,'cos':cos})
    if 'mid' in entry:
        n=entry['native'];epoch=n.lib.lm_mid_checkpoint_reserve(n.mapped)
        if not epoch:raise RuntimeError(n.lib.lm_error().decode())
        checkpoint_tiles=max(1,(length//2)//entry['block'])
        gqa_decode[(32,)](q,k,v,entry['z'],length,entry['mid'],entry['meta'],epoch,
                          checkpoint_tiles,entry['bell'],CHECKPOINT=True,NOTIFY=entry['bell'] is not None,BLOCK=entry['block'],
                          num_warps=entry['warps'],enable_fp_fusion=False)
        if entry['midpoint']:
            n.check((n.lib.lp_control_arm if entry['ready_scheduler'] else n.lib.lp_control_submit)(n.ctx,3,1))
    else:
        if entry['semantic_projection']:
            from python.attention_decode_kernels import partial_output,add_output
            main=torch.cuda.current_stream()
            events={}
            def timed(name,fn):
                if entry['diagnostic']:
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                fn()
                if entry['diagnostic']:
                    end.record();events[name]=(start,end)
            prefix_args=({'Mid':entry['prefix_z'],'CHECK_TILES':max(1,(length//2)//entry['block']),
                          'PREFIX_ONLY':True} if 'prefix_z' in entry else {})
            timed('gqa_a',lambda:gqa_decode[(16,)](q,k,v,entry['z'],length,HEAD_START=0,
                  BLOCK=entry['block'],num_warps=entry['warps'],enable_fp_fusion=False,**prefix_args))
            def proj_a():partial_output[(256,)](entry['z'],entry['weight'],entry['pa'],2048,0,
                                                W_STRIDE=4096,W_START=0,num_warps=4,enable_fp_fusion=False)
            def proj_b():partial_output[(256,)](entry['z'],entry['weight'],entry['pb'],2048,2048,
                                                W_STRIDE=4096,W_START=2048,num_warps=4,enable_fp_fusion=False)
            def mid_norm():
                m=entry['mlp'];n=entry['native']
                norm=m.runtime.model.model.layers[self.layer_idx].post_attention_layernorm.original
                n.check(n.lib.lm_mid_norm(n.mapped,m.input_source.data_ptr(),entry['pa'].data_ptr(),
                                          norm.weight.data_ptr(),norm.variance_epsilon,
                                          torch.cuda.current_stream().cuda_stream))
            if entry['projection_overlap']:
                a_done=torch.cuda.Event();a_done.record(main)
                with torch.cuda.stream(entry['mid_stream']):
                    entry['mid_stream'].wait_event(a_done)
                    timed('proj_a',proj_a)
                    if entry['semantic_mid_publish']:timed('mid_norm',mid_norm)
                    pa_done=torch.cuda.Event();pa_done.record(entry['mid_stream'])
                timed('gqa_b',lambda:gqa_decode[(16,)](q,k,v,entry['z'],length,HEAD_START=16,
                      BLOCK=entry['block'],num_warps=entry['warps'],enable_fp_fusion=False,**prefix_args))
                timed('proj_b',proj_b)
                main.wait_event(pa_done)
            else:
                timed('proj_a',proj_a)
                if entry['semantic_mid_publish']:
                    timed('mid_norm',mid_norm)
                    if entry['semantic_mid_io']:
                        n=entry['native']
                        n.check(n.lib.lp_control_arm(n.ctx,3,1))
                timed('gqa_b',lambda:gqa_decode[(16,)](q,k,v,entry['z'],length,HEAD_START=16,
                      BLOCK=entry['block'],num_warps=entry['warps'],enable_fp_fusion=False,**prefix_args))
                timed('proj_b',proj_b)
            timed('add',lambda:add_output[(16,)](entry['pa'],entry['pb'],entry['y'],num_warps=4))
            entry['phase_events']={**prefix_events,**events}
            result=entry['y']
        elif entry['head_split16']:
            entry['phase_events']=[]
            for first in (0,16):
                if entry['diagnostic']:
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                gqa_decode[(16,)](q,k,v,entry['z'],length,HEAD_START=first,BLOCK=entry['block'],
                                  num_warps=entry['warps'],enable_fp_fusion=False)
                if entry['diagnostic']:
                    end.record();entry['phase_events'].append((start,end))
        else:
            if entry['diagnostic']:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
            gqa_decode[(32,)](q,k,v,entry['z'],length,BLOCK=entry['block'],
                              num_warps=entry['warps'],enable_fp_fusion=False)
            if entry['diagnostic']:
                end.record();entry['phase_events']={**prefix_events,'gqa':(start,end)}
    if not entry['semantic_projection']:
        if entry['diagnostic']:
            result=prefix_timed('o_proj',lambda:self.o_proj(entry['z']))
            entry['phase_events'].update(prefix_events)
        else:result=self.o_proj(entry['z'])
    if entry['capture_output']:entry['last_output']=result
    return result,None,past_key_value
