"""Measure one shared teacher-forced decode workload, sequentially by method."""
import argparse,json,subprocess,sys
from pathlib import Path
if __package__:
    from .common import HERE,PROJECT,load,write,environment
    from .assets import verify
else:
    from common import HERE,PROJECT,load,write,environment
    from assets import verify

def prepare_input(source,tokenizer):
    data=load(source)
    if not all(isinstance(data.get(k),str) and data[k].strip() for k in ['prompt_text','continuation_text']):raise ValueError('Nonempty prompt_text and continuation_text required')
    prompt=tokenizer.encode(data['prompt_text'],add_special_tokens=True).ids
    continuation=tokenizer.encode(data['continuation_text'],add_special_tokens=False).ids
    if not 100<=len(continuation)<=128 or 2 in continuation:raise ValueError('Reference must contain 100..128 non-EOS tokens; no truncation or substitution')
    prefixes=[tokenizer.decode(continuation[:i],skip_special_tokens=True) for i in range(len(continuation)+1)]
    if prefixes[-1]!=data['continuation_text']:raise ValueError('Reference text does not round-trip through the exact tokenizer')
    return dict(prompt_text=data['prompt_text'],prompt_input_ids=prompt,continuation_text=data['continuation_text'],continuation_token_ids=continuation,decoded_prefixes=prefixes,tokenization='Pinned compatibility tokenizer; raw prompt with BOS, continuation without BOS; no chat template',source=data.get('source','User-supplied fixed reference text; not a model-generated answer'))

def main():
    p=argparse.ArgumentParser()
    for name in ['assets','model','build','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--input-file',type=Path,default=PROJECT/'docs/media/fixed_decode_input.json');a=p.parse_args()
    verify(a.assets,model_root=a.model,scope='runtime')
    from tokenizers import Tokenizer
    workload=prepare_input(a.input_file,Tokenizer.from_file(str(a.assets/'metadata/tokenizer_compat.json')))
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'input.json',workload);runs=[]
    for method in ['block','one-shot','leanstream']:
        cmd=[sys.executable,str(HERE/'fixed_token_worker.py'),'--assets',str(a.assets.resolve()),'--model',str(a.model.resolve()),'--build',str(a.build.resolve()),'--input',str((a.output/'input.json').resolve()),'--method',method,'--output',str((a.output/method).resolve())]
        write(a.output/(method+'-command.json'),dict(argv=cmd))
        with (a.output/(method+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment(a.build),check=True)
        runs.append(load(a.output/method/'trace.json'))
    result=dict(format_version=1,kind='fixed_token_decode_trace',model='Ministral-3 8B',device='Jetson AGX Orin',independent_runs=True,run_order=['block','one-shot','leanstream'],workload=workload,records=runs,timing='Monotonic perf_counter_ns from decode start; each forced input token is revealed after its model forward and existing logits.argmax().item() completion boundary. Prefill, warmup, cold binding, reset and final drain excluded. No additional per-step CUDA synchronization.',interpretation='Teacher-forced reference text, not independently generated answers. Single fixed workload, not the historical pooled free-generation benchmark.')
    write(a.output/'fixed_token_decode_trace.json',result)
    print('PASS: all three independent methods processed the identical fixed token sequence')
if __name__=='__main__':main()
