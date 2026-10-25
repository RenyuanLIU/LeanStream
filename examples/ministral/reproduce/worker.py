"""One isolated model process using public components and historical application policies."""
import argparse,csv,hashlib,json,math,os,sys,time
from pathlib import Path
if __package__:
    from .common import bootstrap,load,rows,write,record_lines,sha,provenance,INDICES
else:
    from common import bootstrap,load,rows,write,record_lines,sha,provenance,INDICES

def install(a,model):
    import ctypes as C
    from python.attention_refine import DirectGQAAttention
    if a.method=='leanstream':
        from python.split_async import Runtime
        r=Runtime(model,str(a.assets/'leanstream_weights'),str(a.build/'libpredictor.so'),str(a.build/'libmapped.so'),
            model_spec=dict(H=4096,I=14336,layers=34,active=1434,keep=1219),predictor_root=str(a.assets/'full_predictors'),predictor_dim=1664,final_qd=24,spec_p1_size=1024,ready_scheduler=True,ready_notification='doorbell',p0_budget=32,p1_lite_root=str(a.assets/'p1_lite_predictors'),p1_lite_features=512,p1_lite_budget=128,no_waste_policy='knee',diagnostic=False)
        DirectGQAAttention(model,runtime=r)
        for m in r.layers:
            f=m.native.lib.lp_direct;f.argtypes=[C.c_void_p,C.c_int];f.restype=C.c_int;m.native.check(f(m.native.ctx,512))
    else:
        from baselines import block,one_shot
        r=(block if a.method=='block' else one_shot).install(model,a.assets/'baseline_weights',a.build/'libfixed_cache.so',sanity=a.sanity)
        DirectGQAAttention(model)
    return r

def reset(r,method):
    if method=='leanstream':r.reset()
    else:r.decode=False;r.collect=False

def drain(r,method):
    if method=='leanstream':r.drain()

def bind(r,out):
    from python.leanstream_cuda_backend import bind_runtime
    manifest=bind_runtime(r)
    if len(manifest)!=102 or any(k['grid']!=[610,32,128][k['stage']] or k['warps_per_cta']!=4 for k in manifest):
        raise RuntimeError('Production native geometry must be A610/B32/reduction128, four warps per CTA')
    write(out/'native_kernel_manifest.json',manifest)

def speed(a,model,r,torch):
    prompts=rows(a.assets/'benchmarks/speed_prompts.jsonl');indices=INDICES if a.direction=='forward' else list(reversed(INDICES))
    token_rows=[];generated=[];hot_before=None
    if a.method!='leanstream':
        from baselines.fixed_cache import cache_hash
        hot_before=cache_hash(r.layers)
    def run(index,measured,count=16):
        reset(r,a.method)
        y=model(input_ids=torch.tensor([prompts[index]['input_ids']],device='cuda'),use_cache=True,return_dict=True)
        cache=y.past_key_values;token=int(y.logits[:,-1].argmax(-1).item());ids=[token]
        if a.method!='leanstream':r.decode=True;r.collect=measured;r.sample_id=prompts[index]['sample_id']
        for j in range(count):
            if token==2:break
            if a.method!='leanstream':r.token_index=j
            begin=time.perf_counter_ns()
            y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True,return_dict=True)
            cache=y.past_key_values;token=int(y.logits[:,-1].argmax(-1).item());end=time.perf_counter_ns()
            ids.append(token)
            if measured:token_rows.append(dict(index=index,sample_id=prompts[index]['sample_id'],token_index=j,token_wall_ms=(end-begin)/1e6))
        drain(r,a.method)
        if measured:generated.append(dict(index=index,sample_id=prompts[index]['sample_id'],generated_ids=ids))
    with torch.inference_mode():
        if a.sanity:
            if a.method=='leanstream':raise ValueError('Use the existing numerical.py for LeanStream numerical checks')
            run(0,True,2)
        else:
            run(indices[0],False)
            if a.method=='leanstream':bind(r,a.output)
            proof=provenance(a.assets,a.build,a.method);proof.update(direction=a.direction,cache_policy='dynamic REPAIRED_KNEE' if a.method=='leanstream' else 'immutable fixed-hot 2437 individual neurons',gpu_geometry=dict(A=610,B=32,reduction=128,warps=4) if a.method=='leanstream' else dict(sparse=717,reduction=128,warps=4))
            write(a.output/'run_manifest.json',proof)
            for index in indices:run(index,True)
    if not token_rows:raise RuntimeError('No measured decode calls')
    if a.method!='leanstream':
        from baselines.fixed_cache import cache_hash
        if cache_hash(r.layers)!=hot_before or hot_before!=r.hot_hash:raise RuntimeError('Fixed-hot membership changed')
        if not r.counters or any(x['selected_neurons']!=2867 or x['actual_physical_bytes_read']!=x['SSD_misses']*6144 for x in r.counters):raise RuntimeError('Baseline SSD accounting mismatch')
        if sum(x['actual_physical_bytes_read'] for x in r.counters)<=0:raise RuntimeError('No actual baseline SSD misses executed')
        record_lines(a.output/'io_counters.jsonl',r.counters)
        write(a.output/'fixed_hot_audit.json',dict(before=hot_before,after=cache_hash(r.layers),capacity=2437,selected=2867,immutable=True,bytes_read=sum(x['actual_physical_bytes_read'] for x in r.counters),sanity_checks=r.audit))
        if a.sanity:write(a.output/'run_manifest.json',provenance(a.assets,a.build,a.method))
    with (a.output/'per_token.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(token_rows[0]));writer.writeheader();writer.writerows(token_rows)
    total=sum(x['token_wall_ms'] for x in token_rows)
    write(a.output/'summary.json',dict(status='PASS',method=a.method,direction=a.direction,decode_tokens=len(token_rows),total_decode_ms=total,ms_per_token=total/len(token_rows),tokens_per_s=1000*len(token_rows)/total,sanity=a.sanity,protocol='forward/reverse independent processes, one warmup, EOS stop, up to16 decode calls, timer through logits.item',indices=indices))
    record_lines(a.output/'generated_tokens.jsonl',generated)

def score(a,prompts,tokens):
    from tokenizers import Tokenizer
    if __package__:
        from .scoring.coqa import process_results,doc_to_target
    else:
        from scoring.coqa import process_results,doc_to_target
    docs=load(a.assets/'benchmarks/coqa_docs.json');tokenizer=Tokenizer.from_file(str(a.assets/'metadata/tokenizer_compat.json'))
    decoded=[];scored=[]
    for i,row in enumerate(tokens):
        if row['index']!=i or row['sample_id']!=docs[i]['id'] or row['sample_id']!=prompts[i]['sample_id']:raise ValueError('Scoring record order mismatch')
        prefix=[]
        for step,token in enumerate(row['generated_ids']):
            prefix.append(token);answer=tokenizer.decode(prefix,skip_special_tokens=True)
            if token==2 or '\nQ:' in answer or step+1==64:break
        else:raise ValueError('Generation ended before historical stop')
        answer=answer.split('\nQ:')[0];s=process_results(docs[i],[answer])
        if not all(math.isfinite(s[k]) and 0<=s[k]<=1 for k in ['f1','em']):raise ValueError('Invalid score')
        decoded.append(dict(index=i,sample_id=row['sample_id'],answer=answer,reference_answers=doc_to_target(docs[i])))
        scored.append(dict(index=i,sample_id=row['sample_id'],**s))
    record_lines(a.output/'decoded_answers.jsonl',decoded);record_lines(a.output/'per_sample_accuracy.jsonl',scored)
    f1=sum(x['f1'] for x in scored)/len(scored);em=sum(x['em'] for x in scored)/len(scored)
    write(a.output/'accuracy_summary.json',dict(status='COMPLETE' if len(scored)==500 else 'DRY_RUN_COMPLETE',n=len(scored),f1=f1,em=em,historical_progressive=dict(f1=.7843212061748897,em=.6231666666666666,same_session=False),delta_percentage_points=dict(f1=100*(f1-.7843212061748897),em=100*(em-.6231666666666666)) if len(scored)==500 else None,validity=dict(aligned=True,missing=0,original_scorer=True,finite_logits_checked=True)))

def validate_token_records(tokens,prompts,key,vocab_size,limit):
    if len(tokens)>limit:raise ValueError('Too many resume records')
    for i,row in enumerate(tokens):
        if not isinstance(row,dict) or row.get('index')!=i or row.get('sample_id')!=prompts[i]['sample_id'] or row.get('identity')!=key:
            raise ValueError('Resume records do not match fixed ordering/identity')
        ids=row.get('generated_ids')
        if not isinstance(ids,list) or not 1<=len(ids)<=64 or any(type(v)!=int or not 0<=v<vocab_size for v in ids):
            raise ValueError('Invalid generated token record')
        if 2 in ids[:-1] or (ids[-1]!=2 and len(ids)!=64):
            raise ValueError('Resume record violates historical EOS/length stopping')

def accuracy(a,model,r,torch):
    prompts=rows(a.assets/'benchmarks/coqa_fixed500_prompts.jsonl')
    with torch.inference_mode():
        reset(r,'leanstream');y=model(input_ids=torch.tensor([prompts[0]['input_ids']],device='cuda'),use_cache=True,return_dict=True)
        for _ in range(4):
            token=y.logits[:,-1].argmax(-1).item();y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=y.past_key_values,use_cache=True,return_dict=True)
        r.drain();bind(r,a.output)
        proof=provenance(a.assets,a.build,'leanstream');proof.update(gpu_geometry=dict(A=610,B=32,reduction=128,warps=4),seed=42,max_new_tokens=64,eos_id=2,limit=a.limit,cache_policy='dynamic REPAIRED_KNEE',scorer_sha256=sha(Path(__file__).parent/'scoring/coqa.py'))
        stable=dict(source=proof['source_sha256'],native=proof['native_libraries'],assets=proof['artifact_manifest_sha256'],limit=a.limit,seed=42,max_new_tokens=64,eos_id=2)
        key=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest();proof['identity']=key
        manifest=a.output/'run_manifest.json'
        if manifest.exists() and load(manifest)['identity']!=key:raise ValueError('Resume source/artifact/native identity mismatch')
        write(manifest,proof)
        path=a.output/'generated_tokens.jsonl'
        if path.exists():
            raw=path.read_bytes()
            if raw and not raw.endswith(b'\n'):
                valid=raw[:raw.rfind(b'\n')+1];saved=[json.loads(x) for x in valid.splitlines()]
                validate_token_records(saved,prompts,key,model.config.vocab_size,a.limit)
                path.write_bytes(valid)
        tokens=rows(path) if path.exists() else []
        validate_token_records(tokens,prompts,key,model.config.vocab_size,a.limit)
        with path.open('a') as f:
            for index in range(len(tokens),a.limit):
                r.reset();y=model(input_ids=torch.tensor([prompts[index]['input_ids']],device='cuda'),use_cache=True,return_dict=True);cache=y.past_key_values;generated=[]
                for _ in range(64):
                    logits=y.logits[:,-1]
                    if not torch.isfinite(logits).all().item():raise FloatingPointError(f'Nonfinite logits at sample {index}')
                    token=int(logits.argmax(-1).item());generated.append(token)
                    if token==2:break
                    y=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True,return_dict=True);cache=y.past_key_values
                r.drain();row=dict(index=index,sample_id=prompts[index]['sample_id'],generated_ids=generated,identity=key)
                f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno());tokens.append(row);print('SAMPLE',index,flush=True)
        score(a,prompts,tokens)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,help='Separate pinned upstream checkpoint directory');p.add_argument('--assets',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['block','one-shot','leanstream'],required=True);p.add_argument('--direction',choices=['forward','reverse'],default='forward');p.add_argument('--accuracy',action='store_true');p.add_argument('--limit',type=int,default=500);p.add_argument('--sanity',action='store_true');a=p.parse_args()
    bootstrap();a.assets=a.assets.resolve();a.build=a.build.resolve();a.output=a.output.resolve()
    if a.accuracy and (a.method!='leanstream' or not 1<=a.limit<=500):raise ValueError('Accuracy is LeanStream-only, limit1..500')
    a.output.mkdir(parents=True,exist_ok=a.accuracy)
    import torch
    torch.set_num_threads(1);torch.manual_seed(42)
    from python.ministral3_text_loader import load_text
    model=load_text(str(getattr(a,'model',None) or a.assets/'model'),str(a.assets/'metadata/rope.json'));r=install(a,model)
    try:
        if a.accuracy:accuracy(a,model,r,torch)
        else:speed(a,model,r,torch)
    finally:
        if a.method!='leanstream':r.lib.fc_destroy(r.ctx)
if __name__=='__main__':main()
