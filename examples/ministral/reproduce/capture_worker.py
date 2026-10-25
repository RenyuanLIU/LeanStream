"""One presentation run using unchanged runtime/model/baseline implementations."""
import argparse,os,time,json,hashlib
from pathlib import Path
if __package__:
    from .common import bootstrap,load,write,sha,provenance,PROJECT,MODEL_REPO,MODEL_REVISION
    from .worker import install,reset,drain,bind
else:
    from common import bootstrap,load,write,sha,provenance,PROJECT,MODEL_REPO,MODEL_REVISION
    from worker import install,reset,drain,bind

def main():
    p=argparse.ArgumentParser()
    for name in ['assets','model','build','output','prompt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['block','one-shot','leanstream'],required=True);p.add_argument('--max-new-tokens',type=int,required=True);a=p.parse_args();a.sanity=False
    bootstrap();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from tokenizers import Tokenizer
    from python.ministral3_text_loader import load_text
    torch.set_num_threads(1);torch.manual_seed(42)
    prompt=load(a.prompt);tokenizer=Tokenizer.from_file(str(a.assets/'metadata/tokenizer_compat.json'))
    model=load_text(str(a.model),str(a.assets/'metadata/rope.json'));r=install(a,model)
    hot_before=None
    if a.method!='leanstream':
        from baselines.fixed_cache import cache_hash
        hot_before=cache_hash(r.layers)
    with torch.inference_mode():
        # Original bounded warmup/cold-binding procedure, outside presentation timer.
        reset(r,a.method);y=model(input_ids=torch.tensor([prompt['prompt_input_ids']],device='cuda'),use_cache=True,return_dict=True)
        if a.method!='leanstream':r.decode=True
        for _ in range(4):
            token=int(y.logits[:,-1].argmax(-1).item());y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=y.past_key_values,use_cache=True,return_dict=True)
        drain(r,a.method)
        if a.method=='leanstream':bind(r,a.output)
        proof=provenance(a.assets,a.build,a.method);write(a.output/'run_manifest.json',proof)
        reset(r,a.method);prefill_start=time.perf_counter_ns()
        y=model(input_ids=torch.tensor([prompt['prompt_input_ids']],device='cuda'),use_cache=True,return_dict=True)
        token=int(y.logits[:,-1].argmax(-1).item());cache=y.past_key_values;prefill_end=time.perf_counter_ns();first=token
        if a.method!='leanstream':r.decode=True;r.collect=True;r.sample_id='author-written-ssd-explanation'
        ids=[];completion=[];decode_start=time.perf_counter_ns()
        for j in range(a.max_new_tokens-1):
            if token==2:break
            if a.method!='leanstream':r.token_index=j
            y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True,return_dict=True)
            cache=y.past_key_values;token=int(y.logits[:,-1].argmax(-1).item());end=time.perf_counter_ns()
            ids.append(token);completion.append((end-decode_start)/1e6)
        decode_end=time.perf_counter_ns();drain(r,a.method);drain_end=time.perf_counter_ns()
    if a.method!='leanstream':
        if cache_hash(r.layers)!=hot_before:raise RuntimeError('Fixed-hot cache changed')
        if not r.counters or sum(c['actual_physical_bytes_read'] for c in r.counters)<=0:raise RuntimeError('No real baseline SSD reads')
        io=dict(fixed_hot_immutable=True,selected_neurons=2867,resident_neurons=2437,physical_bytes_read=sum(c['actual_physical_bytes_read'] for c in r.counters))
    else:io=dict(policy='REPAIRED_KNEE',active_groups=1434,keep_groups=1219)
    prefix_ids=[first]+ids
    prefixes=[tokenizer.decode(prefix_ids[:i],skip_special_tokens=True) for i in range(1,len(prefix_ids)+1)]
    native={Path(name).name:digest for name,digest in proof['native_libraries'].items()}
    origins={name:str(Path(path).relative_to(PROJECT)) for name,path in proof['modules'].items()}
    record=dict(method=a.method,prompt_input_ids_sha256=hashlib.sha256(json.dumps(prompt['prompt_input_ids'],separators=(',',':')).encode()).hexdigest(),decoded_prefixes_sha256=hashlib.sha256(json.dumps(prefixes,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),prefill_token_id=first,generated_token_ids=ids,token_completion_ms=completion,decoded_prefixes=prefixes,eos=(prefix_ids[-1]==2),max_new_tokens=a.max_new_tokens,prefill_ms=(prefill_end-prefill_start)/1e6,decode_wall_ms=(decode_end-decode_start)/1e6,final_drain_ms=(drain_end-decode_end)/1e6,model_repository=MODEL_REPO,model_revision=MODEL_REVISION,tokenizer_sha256=sha(a.assets/'metadata/tokenizer_compat.json'),model_config_sha256=sha(a.model/'config.json'),packed_manifest_sha256=sha(a.assets/'leanstream_weights/g2_weights/manifest.json'),predictor_capture_sha256=sha(a.assets/'metadata/predictor_capture.json'),public_source_base=proof['public_commit'],runtime_module_origins=origins,native_library_sha256=native,hardware=dict(machine=proof['hardware']['machine'],cuda=proof['cuda_device']),software=proof['software'],configuration=dict(greedy=True,seed=42,eos_id=2,QD=24,attention='unsplit Direct-GQA',cache_policy='REPAIRED_KNEE' if a.method=='leanstream' else 'immutable fixed-hot',native_pre_enqueue=a.method=='leanstream',B_CTAs=32 if a.method=='leanstream' else None,warps_per_CTA=4),io=io)
    if a.method=='leanstream':record['native_geometry']=[dict(layer=k['layer'],stage=k['stage'],grid=k['grid'],warps=k['warps_per_cta']) for k in load(a.output/'native_kernel_manifest.json')]
    write(a.output/'trace.json',record)
    if a.method!='leanstream':r.lib.fc_destroy(r.ctx)
    print('PASS',a.method,'real decode tokens',len(ids),'milliseconds',completion[-1] if completion else 0,flush=True)
if __name__=='__main__':main()
