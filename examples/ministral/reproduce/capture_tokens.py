"""Bounded, sequential real-token presentation capture; separate from benchmarks."""
import argparse,json,subprocess,sys
from pathlib import Path
if __package__:
    from .common import HERE,load,write,environment
    from .assets import verify
else:
    from common import HERE,load,write,environment
    from assets import verify

PROMPT='Explain how asynchronous SSD weight streaming helps a small computer run a large language model. Write three clear paragraphs for a student.\n\nAnswer:'

def read_prompt(path=None):
    text=PROMPT if path is None else Path(path).read_text(encoding="utf-8")
    if not text.strip():raise ValueError("Prompt file must contain nonempty UTF-8 text")
    return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-new-tokens',type=int,default=96);p.add_argument('--prompt-file',type=Path,help='Author-written UTF-8 prompt; raw text with BOS, no chat template (default unchanged)');a=p.parse_args()
    try:prompt_text=read_prompt(a.prompt_file)
    except (OSError,UnicodeError,ValueError) as exc:p.error(str(exc))
    if not 2<=a.max_new_tokens<=128:p.error('Bounded capture requires 2..128 new tokens')
    verify(a.assets,model_root=a.model,scope='runtime');a.output.mkdir(parents=True,exist_ok=False)
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(a.assets/'metadata/tokenizer_compat.json'))
    prompt=dict(prompt_text=prompt_text,prompt_input_ids=tokenizer.encode(prompt_text,add_special_tokens=True).ids,tokenization='raw author-written English prompt; pinned tokenizer with BOS, no chat template')
    write(a.output/'prompt.json',prompt);runs=[]
    for method in ['block','one-shot','leanstream']:
        out=a.output/method
        cmd=[sys.executable,str(HERE/'capture_worker.py'),'--assets',str(a.assets.resolve()),'--model',str(a.model.resolve()),'--build',str(a.build.resolve()),'--prompt',str((a.output/'prompt.json').resolve()),'--method',method,'--max-new-tokens',str(a.max_new_tokens),'--output',str(out.resolve())]
        write(a.output/(method+'-command.json'),dict(argv=cmd))
        with (a.output/(method+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=environment(a.build),check=True)
        runs.append(load(out/'trace.json'))
    result=dict(format_version=1,kind='real_token_completion_trace',model='Ministral-3 8B',device='Jetson AGX Orin',prompt=prompt,independent_runs=True,run_order=['block','one-shot','leanstream'],timing='monotonic perf_counter_ns elapsed from decode start through existing logits.argmax().item(); prefill and final drain excluded; no extra per-token GPU synchronization',presentation_only=True,records=runs)
    write(a.output/'token_generation_trace.json',result);print('PASS: three independent real-token captures; no benchmark result substituted')
if __name__=='__main__':main()
