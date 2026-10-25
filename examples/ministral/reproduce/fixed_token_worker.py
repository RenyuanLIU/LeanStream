"""Teacher-forced presentation measurement around unchanged production methods."""
import argparse,hashlib,json,time
from pathlib import Path
if __package__:
    from .common import bootstrap,load,write,sha,provenance,PROJECT,MODEL_REPO,MODEL_REVISION
    from .worker import install,reset,drain,bind
else:
    from common import bootstrap,load,write,sha,provenance,PROJECT,MODEL_REPO,MODEL_REVISION
    from worker import install,reset,drain,bind

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ['assets','model','build','output','input']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['block','one-shot','leanstream'],required=True);a=p.parse_args();a.sanity=False
    bootstrap();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from tokenizers import Tokenizer
    from python.ministral3_text_loader import load_text
    torch.set_num_threads(1);torch.manual_seed(42)
    workload=load(a.input);tokens=workload['continuation_token_ids'];prompt=workload['prompt_input_ids']
    tokenizer=Tokenizer.from_file(str(a.assets/'metadata/tokenizer_compat.json'))
    if tokenizer.encode(workload['prompt_text'],add_special_tokens=True).ids!=prompt or tokenizer.encode(workload['continuation_text'],add_special_tokens=False).ids!=tokens:raise ValueError('Workload/tokenizer identity mismatch')
    if not 100<=len(tokens)<=128 or 2 in tokens:raise ValueError('Invalid fixed decode length or EOS in forced inputs')
    model=load_text(str(a.model),str(a.assets/'metadata/rope.json'));r=install(a,model)
    hot_before=None
    if a.method!='leanstream':
        from baselines.fixed_cache import cache_hash
        hot_before=cache_hash(r.layers)
    with torch.inference_mode():
        # Preserve the existing four-call greedy warmup and cold binding outside timing.
        reset(r,a.method);y=model(input_ids=torch.tensor([prompt],device='cuda'),use_cache=True,return_dict=True)
        if a.method!='leanstream':r.decode=True
        for _ in range(4):
            token=int(y.logits[:,-1].argmax(-1).item());y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=y.past_key_values,use_cache=True,return_dict=True)
        drain(r,a.method)
        if a.method=='leanstream':bind(r,a.output)
        proof=provenance(a.assets,a.build,a.method);write(a.output/'run_manifest.json',proof)
        reset(r,a.method);prefill_start=time.perf_counter_ns()
        y=model(input_ids=torch.tensor([prompt],device='cuda'),use_cache=True,return_dict=True)
        int(y.logits[:,-1].argmax(-1).item());cache=y.past_key_values;prefill_end=time.perf_counter_ns()
        if a.method!='leanstream':r.decode=True;r.collect=True;r.sample_id='fixed-token-reference'
        processed=[];predicted=[];completion=[];decode_start=time.perf_counter_ns()
        for j,token in enumerate(tokens):
            if a.method!='leanstream':r.token_index=j
            y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True,return_dict=True)
            cache=y.past_key_values
            # Existing host-visible logits boundary; prediction is recorded but NEVER fed back.
            next_token=int(y.logits[:,-1].argmax(-1).item());end=time.perf_counter_ns()
            completion.append((end-decode_start)/1e6);processed.append(token);predicted.append(next_token)
        decode_end=time.perf_counter_ns();drain(r,a.method);drain_end=time.perf_counter_ns()
    if processed!=tokens or len(completion)!=len(tokens):raise RuntimeError('Not all fixed input tokens were executed')
    if a.method!='leanstream':
        if cache_hash(r.layers)!=hot_before or len(r.counters)!=34*len(tokens):raise RuntimeError('Fixed-hot cache changed or a layer/decode step was skipped')
        if not r.counters or any(c['selected_neurons']!=2867 or c['actual_physical_bytes_read']!=6144*c['SSD_misses'] for c in r.counters):raise RuntimeError('Baseline SSD accounting mismatch')
        io=dict(fixed_hot_immutable=True,selected_neurons=2867,resident_neurons=2437,layer_decode_calls=len(r.counters),physical_bytes_read=sum(c['actual_physical_bytes_read'] for c in r.counters),read_requests=sum(c['read_request_count'] for c in r.counters))
        if io['physical_bytes_read']<=0:raise RuntimeError('No actual SSD reads')
    else:io=dict(policy='REPAIRED_KNEE',active_groups=1434,keep_groups=1219)
    total=completion[-1]
    record=dict(method=a.method,processed_token_ids=processed,processed_token_ids_sha256=digest(processed),prompt_input_ids_sha256=digest(prompt),predicted_next_token_ids=predicted,token_completion_ms=completion,decode_steps=len(processed),decode_wall_ms=(decode_end-decode_start)/1e6,measured_decode_ms=total,ms_per_token=total/len(processed),tokens_per_s=1000*len(processed)/total,prefill_ms=(prefill_end-prefill_start)/1e6,final_drain_ms=(drain_end-decode_end)/1e6,model_repository=MODEL_REPO,model_revision=MODEL_REVISION,tokenizer_sha256=sha(a.assets/'metadata/tokenizer_compat.json'),model_config_sha256=sha(a.model/'config.json'),packed_manifest_sha256=sha(a.assets/'leanstream_weights/g2_weights/manifest.json'),predictor_capture_sha256=sha(a.assets/'metadata/predictor_capture.json'),public_source_base=proof['public_commit'],runtime_module_origins={name:str(Path(path).relative_to(PROJECT)) for name,path in proof['modules'].items()},native_library_sha256={Path(name).name:digest for name,digest in proof['native_libraries'].items()},hardware=dict(machine=proof['hardware']['machine'],cuda=proof['cuda_device']),software=proof['software'],configuration=dict(teacher_forced=True,predicted_tokens_fed_back=False,seed=42,QD=24,attention='unsplit Direct-GQA',cache_policy='REPAIRED_KNEE' if a.method=='leanstream' else 'immutable fixed-hot',native_pre_enqueue=a.method=='leanstream',B_CTAs=32 if a.method=='leanstream' else None,warps_per_CTA=4),io=io)
    if a.method=='leanstream':record['native_geometry']=[dict(layer=k['layer'],stage=k['stage'],grid=k['grid'],warps=k['warps_per_cta']) for k in load(a.output/'native_kernel_manifest.json')]
    write(a.output/'trace.json',record)
    if a.method!='leanstream':r.lib.fc_destroy(r.ctx)
    print('PASS',a.method,len(processed),'fixed decode steps',total,'ms',flush=True)
if __name__=='__main__':main()
