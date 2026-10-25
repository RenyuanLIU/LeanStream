"""Replay actual autoregressive outputs; never substitutes tokens or timestamps."""
import argparse,bisect,hashlib,importlib.util,json,math,textwrap
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(p.read_text())
def assemble(results,prompt_id,tokenizer_path,prefix):
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(tokenizer_path));records=[];common=None
 variants=['block','calibrated1638','leanstream']
 for method,v in zip(['block','one-shot','leanstream'],variants):
  folder=results/(prefix+v);rows=[json.loads(x) for x in (folder/'generation.jsonl').read_text().splitlines()];matches=[x for x in rows if x['prompt_id']==prompt_id];assert len(matches)==1
  row=matches[0];manifest=load(folder/'manifest.json');ids=row['generated_ids'];assert row['feedback_input_ids']==ids[:-1]
  assert len(row['completion_ms'])==len(ids)-1==row['decode_calls'];assert tok.decode(ids,skip_special_tokens=True)==row['decoded_text']
  assert tok.encode(row['prompt_text'],add_special_tokens=True).ids==row['input_ids']
  prompt=dict(prompt_text=row['prompt_text'],prompt_input_ids=row['input_ids'])
  if common is None:common=prompt
  assert common==prompt
  if 'kv_cache_lengths' in row:assert row['kv_cache_lengths']==list(range(len(row['input_ids'])+1,len(row['input_ids'])+len(ids)))
  prefixes=[tok.decode(ids[:i],skip_special_tokens=True) for i in range(1,len(ids)+1)]
  rec=dict(method=method,variant=v,tokenizer_sha256=sha(tokenizer_path),model_config_sha256=manifest['model']['files']['config.json']['sha256'],model_revision=manifest['model']['revision'],prompt_input_ids_sha256=digest(row['input_ids']),max_new_tokens=64,prefill_token_id=ids[0],generated_token_ids=ids[1:],token_completion_ms=row['completion_ms'],decoded_prefixes=prefixes,decoded_prefixes_sha256=digest(prefixes),eos=row['eos'],decode_wall_ms=row['decode_ms'],decode_calls=row['decode_calls'],actual_generated_ids=ids,feedback_input_ids=row['feedback_input_ids'],decoded_text=row['decoded_text'],measured_decode_ms_per_token=row['decode_ms']/row['decode_calls'] if row['decode_calls'] else None,measured_decode_tokens_per_s=1000*row['decode_calls']/row['decode_ms'] if row['decode_calls'] else None,source_generation_sha256=sha(folder/'generation.jsonl'),source_manifest_sha256=sha(folder/'manifest.json'),experimental_predictor_manifest_sha256=manifest.get('experimental_predictor_manifest_sha256'))
  if method!='leanstream':
   io=[json.loads(x) for x in (folder/'io.jsonl').read_text().splitlines() if json.loads(x)['sample_id']==prompt_id]
   rec['ssd']=dict(layer_records=len(io),actual_physical_bytes_read=sum(x['actual_physical_bytes_read'] for x in io));assert len(io)==34*row['decode_calls'];assert load(folder/'fixed_hot.json')['immutable']
  records.append(rec)
 data=dict(format_version=1,kind='real_token_completion_trace',independent_runs=True,prompt=common,records=records,experimental=True,description='Real greedy autoregressive runs; recalibrated One-shot experimental. Prefill token at zero; common measured decode time and common replay scale. Different EOS lengths; no claim of resolved general generation quality.')
 util=Path(__file__).resolve().parents[1]/'tools/make_speed_gif.py';spec=importlib.util.spec_from_file_location('existing_renderer',util);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.validate_trace(data,tok)
 return data

def render(data,output,seconds=14,fps=12):
 from PIL import Image,ImageDraw,ImageFont,ImageChops
 fonts={n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',n) for n in [17,18,20,21,22,25,28]}
 duration=max(r['token_completion_ms'][-1] for r in data['records']);frames=[];previous=None;palette=None;output.parent.mkdir(parents=True,exist_ok=True)
 prompt_lines=textwrap.wrap('Prompt: '+data['prompt']['prompt_text'].replace('\n',' '),width=122)
 top=max(184,105+24*len(prompt_lines)+15);bottom=top+430;height=bottom+90
 for f in range(seconds*fps):
  t=duration*min(f/(seconds*fps-18),1);im=Image.new('RGB',(1600,height),'#0b1421');d=ImageDraw.Draw(im)
  d.text((24,16),'Actual autoregressive generation — experimental comparison',font=fonts[28],fill='#edf3fa')
  d.text((24,57),'Ministral-3 8B | Jetson AGX Orin | Greedy decode, maximum 64 tokens',font=fonts[20],fill='#acbcd0')
  d.text((1260,57),f'Decode: {t/1000:5.2f} s',font=fonts[20],fill='#edf3fa')
  for i,line in enumerate(prompt_lines):d.text((24,96+24*i),line,font=fonts[20],fill='#d1dce9')
  for j,(r,name,color) in enumerate(zip(data['records'],['Block','One-shot (Recalibrated)','LeanStream'],['#b9c6d6','#e7c58c','#64d8ba'])):
   left=20+j*526;right=left+510;d.rounded_rectangle((left,top,right,bottom),radius=9,fill='#101f30',outline='#334356',width=2)
   d.text((left+16,top+13),name,font=fonts[22],fill=color)
   label='Experimental early-input predictor' if j==1 else 'Original implementation'
   d.text((left+16,top+46),label,font=fonts[18],fill='#acbcd0')
   d.text((left+16,top+75),f'{r["measured_decode_tokens_per_s"]:.3f} tok/s | {r["measured_decode_ms_per_token"]:.1f} ms/tok',font=fonts[18],fill=color)
   n=bisect.bisect_right(r['token_completion_ms'],t);lines=[]
   for para in r['decoded_prefixes'][n].split('\n'):lines.extend(textwrap.wrap(para,width=39,replace_whitespace=False,drop_whitespace=True) or [''])
   for i,line in enumerate(lines[-10:]):d.text((left+16,top+120+i*26),line,font=fonts[20],fill='#e6edf6')
   state=('EOS' if r['eos'] else 'token cap') if n==len(r['generated_token_ids']) else 'decoding'
   d.text((left+16,bottom-32),f'{n+1} output tokens incl. prefill | {state}',font=fonts[17],fill=color)
  d.text((24,bottom+17),'Same prompt; independent real runs. Each model feeds its own predicted tokens back.',font=fonts[20],fill='#bdccde')
  d.text((24,bottom+48),'Common offline replay scale; prefill token at t=0. Unequal EOS lengths; single-prompt timing, not pooled benchmark.',font=fonts[17],fill='#92a6c0')
  if f==seconds*fps//2:im.save(output.with_name(output.stem+'_midpoint.png'))
  if f==seconds*fps-1:im.save(output.with_name(output.stem+'_final.png'))
  if palette is None:palette=im.quantize(colors=128)
  q=im.quantize(palette=palette,dither=0)
  if previous is None:encoded=q.copy()
  else:
   mask=ImageChops.difference(q,previous).point(lambda v:255 if v==0 else 0,mode='1');encoded=q.copy();encoded.paste(255,mask=mask)
  previous=q.copy();encoded.info['transparency']=255;frames.append(encoded)
 frames[0].save(output,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0,optimize=False,disposal=1,transparency=255)
 # Confirm composited GIF frames, not only the source raster images.
 with Image.open(output) as gif:
  total=0
  for i in range(gif.n_frames):gif.seek(i);total+=gif.info.get('duration',0)
  elapsed=0
  for i in range(gif.n_frames):
   gif.seek(i);elapsed+=gif.info.get('duration',0)
   if elapsed>=total/2:
    gif.convert('RGB').save(output.with_name(output.stem+'_gif_midpoint.png'));break
  gif.seek(gif.n_frames-1);gif.convert('RGB').save(output.with_name(output.stem+'_gif_final.png'))
 return dict(gif=str(output),bytes=output.stat().st_size,duration_ms=duration,frames=len(frames))

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenizer',type=Path);a=p.parse_args()
 data=load(a.trace);tokenizer=None
 if a.tokenizer:
  from tokenizers import Tokenizer
  if sha(a.tokenizer)!=data['records'][0]['tokenizer_sha256']:raise ValueError('Tokenizer hash mismatch')
  tokenizer=Tokenizer.from_file(str(a.tokenizer))
 util=Path(__file__).resolve().parent/'make_speed_gif.py';spec=importlib.util.spec_from_file_location('trace_validator',util);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.validate_trace(data,tokenizer)
 for rec in data['records']:
  calls=len(rec['generated_token_ids']);rec.setdefault('measured_decode_tokens_per_s',1000*calls/rec['decode_wall_ms']);rec.setdefault('measured_decode_ms_per_token',rec['decode_wall_ms']/calls)
 print(json.dumps(render(data,a.output)))
if __name__=='__main__':main()
