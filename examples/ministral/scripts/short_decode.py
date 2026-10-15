"""Bounded application-level decode timing for the validated Ministral knee path.
Prefill, reset and maintenance drain remain outside the per-token timer.
No extra per-token synchronization or layer profiling is enabled.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from python.ministral3_text_loader import load_text
from python.split_async import Runtime
from python.attention_refine import DirectGQAAttention


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = sorted(values)
    if not values:
        return None
    def pct(p):
        x = (len(values)-1)*p
        lo = math.floor(x)
        return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(x-lo)
    return dict(count=len(values), mean=statistics.mean(values), min=values[0],
                p50=pct(.5), p90=pct(.9), p95=pct(.95), max=values[-1])


class LayerProbe:
    def __init__(self, model, enabled, nvtx=False, detailed=False):
        self.enabled=enabled
        self.nvtx=nvtx
        self.detailed=detailed
        self.active=False
        self.sample_id=None
        self.token_index=-1
        self.host_begin=[0]*len(model.model.layers)
        self.host_end=[0]*len(model.model.layers)
        self.start=[torch.cuda.Event(enable_timing=True) for _ in model.model.layers] if enabled else []
        self.end=[torch.cuda.Event(enable_timing=True) for _ in model.model.layers] if enabled else []
        self.module_events={}
        self.handles=[]
        if enabled:
            for i,layer in enumerate(model.model.layers):
                self.handles.append(layer.register_forward_pre_hook(self.before(i),prepend=True))
                self.handles.append(layer.register_forward_hook(self.after(i)))
                if nvtx:
                    self.handles.append(layer.self_attn.register_forward_pre_hook(self.component_before(i,'attention')))
                    self.handles.append(layer.self_attn.register_forward_hook(self.component_after()))
                    self.handles.append(layer.mlp.register_forward_pre_hook(self.component_before(i,'MLP')))
                    self.handles.append(layer.mlp.register_forward_hook(self.component_after()))
                if detailed:
                    for name,module in (('input_rmsnorm',layer.input_layernorm),
                                        ('post_attention_rmsnorm',layer.post_attention_layernorm)):
                        pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                        self.module_events[(i,name)]=pair
                        self.handles.append(module.register_forward_pre_hook(self.module_before(pair)))
                        self.handles.append(module.register_forward_hook(self.module_after(pair)))
            for event in self.start+self.end:
                event.record()
            for pair in self.module_events.values():
                for event in pair:event.record()

    def before(self,i):
        def hook(module, inputs):
            if self.active:
                self.host_begin[i]=time.perf_counter_ns()
                if self.nvtx:
                    torch.cuda.nvtx.range_push(f'sample={self.sample_id} token={self.token_index} layer={i}')
                self.start[i].record(torch.cuda.current_stream())
        return hook

    def after(self,i):
        def hook(module, inputs, output):
            if self.active:
                self.end[i].record(torch.cuda.current_stream())
                self.host_end[i]=time.perf_counter_ns()
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()
        return hook

    def component_before(self,i,name):
        def hook(module, inputs):
            if self.active:
                torch.cuda.nvtx.range_push(f'sample={self.sample_id} token={self.token_index} layer={i} {name}')
        return hook

    def component_after(self):
        def hook(module, inputs, output):
            if self.active:
                torch.cuda.nvtx.range_pop()
        return hook

    def module_before(self,pair):
        def hook(module,inputs):
            if self.active:pair[0].record(torch.cuda.current_stream())
        return hook

    def module_after(self,pair):
        def hook(module,inputs,output):
            if self.active:pair[1].record(torch.cuda.current_stream())
        return hook

    def rows(self):
        # Normal logits .item() has already completed the model stream.
        return [dict(sample_id=self.sample_id, token_index=self.token_index, layer=i,
                     layer_wall_ms=self.start[i].elapsed_time(self.end[i]),
                     layer_cpu_enqueue_ms=(self.host_end[i]-self.host_begin[i])/1e6,
                     layer_gpu_active_ms='')
                for i in range(len(self.start))]


def write_csv(path, rows, names):
    with open(path,'w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    for key in ('model','weights','predictor-root','predictor-library','shared','prompts','output','p1-lite-root'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--rope',required=True)
    p.add_argument('--accuracy-records',default=None,help='Optional saved token records for comparison only')
    p.add_argument('--indices',default='0,2,5,6,7,8,9,11')
    p.add_argument('--max-decode-tokens',type=int,default=16)
    p.add_argument('--warmup-prompts',type=int,default=1)
    p.add_argument('--no-layer-events',action='store_true',default=True)
    p.add_argument('--layer-sample-every',type=int,default=4,
                   help='Record layer CUDA events on every Nth decode token')
    p.add_argument('--nvtx',action='store_true',help='Detailed short Nsight trace only; off for lightweight timing')
    p.add_argument('--detailed-events',action='store_true',help='Short diagnostic CUDA/native timing; no token/layer synchronization')
    a=p.parse_args()
    if a.nvtx or a.detailed_events:raise ValueError("This runner supports application-level timing only")
    if (a.max_decode_tokens<1 or a.max_decode_tokens>64 or a.warmup_prompts<0 or
            a.layer_sample_every<1):
        raise ValueError('Invalid bounded benchmark length')
    expected_profile='2' if a.detailed_events else '0'
    if os.getenv('LEANSTREAM_STREAM_QD')!='24' or os.getenv('LEANSTREAM_STREAM_PROFILE')!=expected_profile:
        raise RuntimeError(f'Expected QD24 and stream profile {expected_profile}')
    indices=[int(i) for i in a.indices.split(',')]
    if len(indices)!=len(set(indices)) or any(i<0 for i in indices):
        raise ValueError('Invalid fixed CoQA indices')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model=load_text(a.model,a.rope)
    spec=dict(H=4096,I=14336,layers=34,active=1434,keep=1219)
    runtime=Runtime(model,a.weights,a.predictor_library,a.shared,model_spec=spec,
                    predictor_root=a.predictor_root,predictor_dim=1664,final_qd=24,
                    spec_p1_size=1024,ready_scheduler=True,ready_notification='doorbell',
                    p0_budget=32,p1_lite_root=a.p1_lite_root,p1_lite_features=512,
                    p1_lite_budget=128,no_waste_policy='knee',diagnostic=a.detailed_events)
    attention=DirectGQAAttention(model,runtime=runtime)
    for layer in runtime.layers:
        lib=layer.native.lib
        import ctypes as C
        lib.lp_direct.argtypes=[C.c_void_p,C.c_int]
        lib.lp_direct.restype=C.c_int
        layer.native.check(lib.lp_direct(layer.native.ctx,512))
    prompts=[json.loads(line) for line in Path(a.prompts).read_text().splitlines()]
    accuracy=[json.loads(line) for line in Path(a.accuracy_records).read_text().splitlines()] if a.accuracy_records else None
    if any(i>=len(prompts) for i in indices) or (accuracy is not None and (len(prompts)!=len(accuracy) or any(prompts[i]['sample_id']!=accuracy[i]['sample_id'] for i in indices))):
        raise RuntimeError('Fixed CoQA record order differs from completed knee accuracy run')
    probe=LayerProbe(model,not a.no_layer_events,a.nvtx,a.detailed_events)
    output=Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    token_rows=[]
    layer_rows=[]
    prompt_rows=[]
    generated_rows=[]
    detailed_gpu=[]
    detailed_cpu=[]
    detailed_io=[]
    def run(index,measured):
        prompt=prompts[index]
        runtime.reset()
        # Reset is outside the generation interval, as in the established runner.
        start=time.perf_counter_ns()
        if measured and a.nvtx:
            torch.cuda.nvtx.range_push(f'sample={prompt["sample_id"]} prefill')
        output_model=model(input_ids=torch.tensor([prompt['input_ids']],device='cuda'),use_cache=True,return_dict=True)
        cache=output_model.past_key_values
        token=int(output_model.logits[:,-1].argmax(-1).item())
        prefill_end=time.perf_counter_ns()
        generated=[token]
        if measured and a.nvtx:
            torch.cuda.nvtx.range_pop()
        decode_begin=prefill_end
        for j in range(a.max_decode_tokens):
            if token==2:
                break
            begin=time.perf_counter_ns()
            if measured and a.nvtx:
                torch.cuda.nvtx.range_push(f'sample={prompt["sample_id"]} token={j}')
            probe.active=measured and probe.enabled and j%a.layer_sample_every==0
            probe.sample_id=prompt['sample_id']
            probe.token_index=j
            output_model=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,
                               use_cache=True,return_dict=True)
            cache=output_model.past_key_values
            token=int(output_model.logits[:,-1].argmax(-1).item())
            generated.append(token)
            end=time.perf_counter_ns()
            sampled_layer=probe.active
            probe.active=False
            if measured:
                if a.nvtx:
                    torch.cuda.nvtx.range_pop()
                token_rows.append(dict(sample_id=prompt['sample_id'],token_index=j,
                                       token_wall_ms=(end-begin)/1e6))
                if sampled_layer:
                    layer_rows.extend(probe.rows())
                if a.detailed_events:
                    import ctypes as C
                    attention_rows=attention.rows()
                    for i,m in enumerate(runtime.layers):
                        events=m.events
                        intervals={name:events[name][0].elapsed_time(events[name][1])
                                   for name in ('attention','norm','A','B_including_wait','reduction',
                                                'publication','packed_cache_update','mapped_importance')}
                        intervals.update({name:pair[0].elapsed_time(pair[1])
                                          for (layer_id,name),pair in probe.module_events.items() if layer_id==i})
                        stamp=m.engine.stamp.host.reshape(m.engine.programs,4)
                        pa=m.engine.pa
                        b=stamp[pa:]
                        intervals['B_exposed_wait_ms']=max(0,int(b[:,1].min())-int(b[:,0].min()))/1e6
                        intervals['B_compute_timestamp_ms']=(int(b[:,3].max())-int(b[:,2].min()))/1e6
                        detailed_gpu.append(dict(sample_id=prompt['sample_id'],token_index=j,
                                                 layer=i,attention_operators_ms=attention_rows[i],
                                                 runtime_spans_ms=intervals,
                                                 kv_length=attention.entries[i].get('last_kv_length')))
                        lib=m.native.lib
                        lib.lp_ready_predictor_trace.argtypes=[C.c_void_p,C.c_int]
                        lib.lp_ready_predictor_trace.restype=C.POINTER(C.c_uint64)
                        traces={stage:[int(lib.lp_ready_predictor_trace(m.native.ctx,stage)[k]) for k in range(6)]
                                for stage in (1,2)}
                        detailed_cpu.append(dict(sample_id=prompt['sample_id'],token_index=j,
                                                 layer=i,ready_predictor_trace_ns=traces,
                                                 p1_ready_to_start_us=(traces[1][2]-traces[1][1])/1000 if traces[1][1] else None,
                                                 p2_ready_to_start_us=(traces[2][2]-traces[2][1])/1000 if traces[2][1] else None,
                                                 p1_predict_us=(traces[1][3]-traces[1][2])/1000 if traces[1][2] else None,
                                                 p2_predict_us=(traces[2][3]-traces[2][2])/1000 if traces[2][2] else None))
        decode_end=time.perf_counter_ns()
        runtime.drain()  # Existing per-prompt drain, after measured decode tokens.
        total_end=time.perf_counter_ns()
        if measured and a.detailed_events:
            import ctypes as C
            raise RuntimeError('Detailed profiling is not supported by this runner')
        reference=accuracy[index]['generated_ids'][:len(generated)] if accuracy is not None else generated[:]
        if measured:
            first_divergence=next((i for i,(got,want) in enumerate(zip(generated,reference)) if got!=want),None)
            if first_divergence is None and len(generated)!=len(reference):
                first_divergence=min(len(generated),len(reference))
            generated_rows.append(dict(index=index,sample_id=prompt['sample_id'],
                                       generated_ids=generated,reference_ids=reference,
                                       prefix_matches=(first_divergence is None) if accuracy is not None else None,
                                       first_divergence_index=first_divergence))
            prompt_rows.append(dict(index=index,sample_id=prompt['sample_id'],
                prefill_ms=(prefill_end-start)/1e6,decode_ms=(decode_end-decode_begin)/1e6,
                final_drain_ms=(total_end-decode_end)/1e6,e2e_ms=(total_end-start)/1e6,
                generated_tokens=len(generated),decode_tokens=len(generated)-1))
    with torch.inference_mode():
        for _ in range(a.warmup_prompts):
            run(indices[0],False)
        if a.nvtx:
            torch.cuda.nvtx.range_push('profile_run')
        for index in indices:
            run(index,True)
        if a.nvtx:
            torch.cuda.nvtx.range_pop()
    write_csv(output/'per_token.csv',token_rows,['sample_id','token_index','token_wall_ms'])
    if a.detailed_events:
        for name,rows in (('detailed_gpu.jsonl',detailed_gpu),('detailed_cpu.jsonl',detailed_cpu),
                          ('detailed_io.jsonl',detailed_io)):
            with (output/name).open('w') as file:
                for row in rows:file.write(json.dumps(row,allow_nan=False)+'\n')
    with (output/'generated_tokens.jsonl').open('w') as file:
        for row in generated_rows:
            file.write(json.dumps(row)+'\n')
    write_csv(output/'per_layer.csv',layer_rows,
              ['sample_id','token_index','layer','layer_wall_ms','layer_cpu_enqueue_ms','layer_gpu_active_ms'])
    layers=[]
    for i in range(34):
        rows=[r for r in layer_rows if r['layer']==i]
        span=stats([r['layer_wall_ms'] for r in rows])
        enqueue=stats([r['layer_cpu_enqueue_ms'] for r in rows])
        layers.append(dict(layer=i,count=len(rows),wall_mean_ms=span['mean'] if span else '',
            wall_p50_ms=span['p50'] if span else '',wall_p90_ms=span['p90'] if span else '',
            wall_p95_ms=span['p95'] if span else '',gpu_mean_ms='',
            cpu_enqueue_mean_ms=enqueue['mean'] if enqueue else ''))
    write_csv(output/'per_layer_summary.csv',layers,
              ['layer','count','wall_mean_ms','wall_p50_ms','wall_p90_ms','wall_p95_ms',
               'gpu_mean_ms','cpu_enqueue_mean_ms'])
    summary=dict(status='PASS',mode='repaired_knee',profile=expected_profile,qd=24,
        timing_definition=dict(e2e='perf_counter_ns from prefill call to final prompt drain',
            prefill='perf_counter_ns through ordinary logits scalar read',
            token='perf_counter_ns through ordinary logits scalar read',
            layer_wall='CUDA event stream span; includes stream waits and launch gaps, not CPU wall or GPU active-kernel time',
            cpu_enqueue='host pre-hook to post-hook; includes host waits, not CPU active time',
            gpu_active='unavailable without tracing; blank in CSV'),
        prompts=len(prompt_rows),decode_tokens=len(token_rows),
        e2e_ms=stats([r['e2e_ms'] for r in prompt_rows]),
        prefill_ms=stats([r['prefill_ms'] for r in prompt_rows]),
        decode_total_ms=stats([r['decode_ms'] for r in prompt_rows]),
        decode_token_ms=stats([r['token_wall_ms'] for r in token_rows]),
        aggregate_decode_tokens_per_s=1000*len(token_rows)/sum(r['decode_ms'] for r in prompt_rows),
        mean_derived_tokens_per_s=1000/statistics.mean(r['token_wall_ms'] for r in token_rows),
        max_decode_tokens=a.max_decode_tokens,indices=indices,layer_events=probe.enabled,
        layer_sample_every=a.layer_sample_every,nvtx=a.nvtx,detailed_events=a.detailed_events,
        matching_saved_accuracy_prefixes=sum(r['prefix_matches'] for r in generated_rows) if accuracy is not None else None,
        model_config_sha256=digest(Path(a.model)/'config.json'),
        packed_manifest_sha256=digest(Path(a.weights)/'g2_weights/manifest.json'),
        prompt_sha256=digest(a.prompts),shared_sha256=digest(a.shared),
        accuracy_records_sha256=digest(a.accuracy_records) if a.accuracy_records else None)
    (output/'profile_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(output/'per_prompt.csv',prompt_rows,
              ['index','sample_id','prefill_ms','decode_ms','final_drain_ms','e2e_ms','generated_tokens','decode_tokens'])
    print(json.dumps(dict(status='PASS',prompts=len(prompt_rows),tokens=len(token_rows),
                          decode_p50_ms=summary['decode_token_ms']['p50'])),flush=True)


if __name__=='__main__':
    main()
