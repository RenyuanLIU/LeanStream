"""Replay measured fixed-token forwards or retained free-generation diagnostics."""
import argparse,bisect,hashlib,json,math,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISION='e3a799844ba08f27fddbf118e154057eb0105889'
METHODS=['block','one-shot','leanstream']

def digest_json(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def validate_trace(data,tokenizer=None):
    if data.get("kind")=="fixed_token_decode_trace":return validate_fixed_trace(data,tokenizer)
    if data.get('format_version')!=1 or data.get('kind')!='real_token_completion_trace' or data.get('independent_runs') is not True:raise ValueError('Real independent token traces required; aggregate-rate traces are not accepted')
    prompt=data['prompt'];ids=prompt['prompt_input_ids']
    if not prompt['prompt_text'] or not ids or any(type(x)!=int or x<0 for x in ids):raise ValueError('Invalid common real prompt')
    runs=data['records']
    if len(runs)!=3 or [r['method'] for r in runs]!=METHODS:raise ValueError('Exactly Block, One-shot, LeanStream records required')
    if len({r['tokenizer_sha256'] for r in runs})!=1:raise ValueError('Tokenizer identities differ')
    if len({r['model_config_sha256'] for r in runs})!=1 or len({r['max_new_tokens'] for r in runs})!=1:raise ValueError('Model/decode constraints differ')
    for r in runs:
        if r['model_revision']!=REVISION or r['prompt_input_ids_sha256']!=digest_json(ids):raise ValueError('Prompt IDs/model revision differ')
        tokens=r['generated_token_ids'];times=r['token_completion_ms'];prefixes=r['decoded_prefixes'];maximum=r['max_new_tokens']
        if not 2<=maximum<=128 or len(tokens)!=len(times) or len(tokens)>maximum-1 or any(type(x)!=int or x<0 for x in [r['prefill_token_id']]+tokens):raise ValueError('Invalid real token records')
        if any(type(t) not in (float,int) or not math.isfinite(t) or t<=0 for t in times) or any(a>=b for a,b in zip(times,times[1:])):raise ValueError('Completion timestamps must be strictly monotonic')
        if len(prefixes)!=len(tokens)+1 or any(not isinstance(x,str) for x in prefixes) or digest_json(prefixes)!=r['decoded_prefixes_sha256']:raise ValueError('Decoded prefix identity mismatch')
        all_ids=[r['prefill_token_id']]+tokens
        if 2 in all_ids[:-1] or r['eos']!=(all_ids[-1]==2):raise ValueError('Incorrect EOS stopping')
        if times and times[-1]>r['decode_wall_ms']+.01:raise ValueError('Completion exceeds decode interval')
        if tokenizer is not None:
            for i,prefix in enumerate(prefixes,1):
                if tokenizer.decode(all_ids[:i],skip_special_tokens=True)!=prefix:raise ValueError('Text is not the tokenizer-decoded real token prefix')
    return data

def render(data,output,preview=None,seconds=14,fps=12,diagnostic=False):
    if data.get("kind")=="fixed_token_decode_trace":return render_fixed(data,output,preview,seconds,fps)
    from PIL import Image,ImageChops,ImageDraw,ImageFont
    path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
    if not path.is_file():raise ValueError('Install/provide DejaVu Sans Mono for deterministic readable rendering')
    fonts={n:ImageFont.truetype(str(path),n) for n in [17,18,20,22,24,28]}
    runs=data['records'];duration=max((r['token_completion_ms'][-1] if r['token_completion_ms'] else 0) for r in runs)
    if duration<=0:raise ValueError('No decode completions to replay')
    colors=['#b9c6d6','#e7c58c','#64d8ba'];names=['Block','One-shot','LeanStream'];frames=[];palette=None;previous=None
    for frame in range(seconds*fps):
        elapsed=duration*min(frame/(seconds*fps-12),1)
        im=Image.new('RGB',(1600,620),'#0b1421');d=ImageDraw.Draw(im)
        d.text((25,16),'LeanStream — On-Device LLM Inference',font=fonts[28],fill='#edf3fa')
        d.text((25,55),'Ministral-3 8B | Jetson AGX Orin',font=fonts[20],fill='#acbcd0')
        d.text((1240,53),f'Decode: {elapsed/1000:5.1f} s',font=fonts[22],fill='#edf3fa')
        prompt_lines=textwrap.wrap('Prompt: '+data['prompt']['prompt_text'].replace('\n',' '),width=124)
        for i,line in enumerate(prompt_lines):d.text((25,92+24*i),line,font=fonts[20],fill='#d1dce9')
        top=160;bottom=548
        for j,(r,color,name) in enumerate(zip(runs,colors,names)):
            left=20+j*526;right=left+510
            d.rounded_rectangle((left,top,right,bottom),radius=8,fill='#101f30',outline='#334356',width=2)
            d.rectangle((left+2,top+28,right-2,top+52),fill='#1a2b3d')
            d.text((left+17,top+12),name,font=fonts[24],fill=color)
            count=bisect.bisect_right(r['token_completion_ms'],elapsed)
            text=r['decoded_prefixes'][count];lines=[]
            for paragraph in text.split('\n'):
                lines.extend(textwrap.wrap(paragraph,width=35,replace_whitespace=False,drop_whitespace=True) or [''])
            lines=lines[-10:]
            for i,line in enumerate(lines):d.text((left+16,top+67+27*i),line,font=fonts[22],fill='#e6edf6')
            status='EOS' if count==len(r['generated_token_ids']) and r['eos'] else ('limit' if count==len(r['generated_token_ids']) else 'decoding')
            d.text((left+16,bottom-29),f'{count} decode tokens | {status}',font=fonts[17],fill=color)
        d.text((25,566),'Same prompt, independent real runs. Diagnostic replay: One-shot repetition is preserved.' if diagnostic else 'Same prompt, independent real inference runs. Decode traces synchronized for offline replay.',font=fonts[18],fill='#bdccde')
        d.text((25,591),'Common time scale; prefill token at t=0. Different EOS lengths; this is not a throughput comparison.' if diagnostic else 'Common time scale; prefill token visible at t=0. Presentation capture, not the pooled benchmark.',font=fonts[17],fill='#92a6c0')
        if palette is None:palette=im.quantize(colors=128)
        quantized=im.quantize(palette=palette,dither=0)
        if preview and frame==seconds*fps//2:quantized.convert('RGB').save(preview)
        if previous is None:encoded=quantized
        else:
            unchanged=ImageChops.difference(quantized,previous).point(lambda value:255 if value==0 else 0,mode='1')
            encoded=quantized.copy();encoded.paste(255,mask=unchanged)
        previous=quantized.copy();encoded.info['transparency']=255;frames.append(encoded)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0,optimize=False,disposal=1,transparency=255)

def validate_fixed_trace(data,tokenizer=None):
    if data.get('format_version')!=1 or data.get('independent_runs') is not True or data.get('run_order')!=METHODS:raise ValueError('Independent fixed-token runs required')
    w=data['workload'];tokens=w['continuation_token_ids'];prompt=w['prompt_input_ids'];prefixes=w['decoded_prefixes'];count=len(tokens)
    if not 100<=count<=128 or not prompt or any(type(x)!=int or x<0 for x in prompt+tokens) or 2 in tokens:raise ValueError('Invalid fixed input sequence')
    if len(prefixes)!=count+1 or prefixes[0]!='' or prefixes[-1]!=w['continuation_text'] or any(not isinstance(x,str) for x in prefixes):raise ValueError('Invalid fixed reference prefixes')
    runs=data['records']
    if len(runs)!=3 or [r['method'] for r in runs]!=METHODS:raise ValueError('Exactly three original methods required')
    for key in ['tokenizer_sha256','model_config_sha256','packed_manifest_sha256','predictor_capture_sha256']:
        if len({r[key] for r in runs})!=1:raise ValueError('Model/artifact identities differ')
    for r in runs:
        if r['model_revision']!=REVISION or r['processed_token_ids']!=tokens or r['processed_token_ids_sha256']!=digest_json(tokens) or r['prompt_input_ids_sha256']!=digest_json(prompt):raise ValueError('Methods did not process identical pinned inputs')
        times=r['token_completion_ms']
        if r['decode_steps']!=count or len(times)!=count or len(r['predicted_next_token_ids'])!=count:raise ValueError('Decode step count differs')
        if any(type(x)!=int or x<0 for x in r['predicted_next_token_ids']):raise ValueError('Invalid recorded predictions')
        if any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in times) or any(a>=b for a,b in zip(times,times[1:])):raise ValueError('Actual monotonic completion timestamps required')
        if times[-1]>r['decode_wall_ms']+.01:raise ValueError('Completion exceeds decode interval')
        for key,expected in [('measured_decode_ms',times[-1]),('ms_per_token',times[-1]/count),('tokens_per_s',1000*count/times[-1])]:
            if not math.isclose(r[key],expected,rel_tol=1e-12):raise ValueError('Throughput does not match measured timestamps')
        c=r['configuration']
        if c['teacher_forced'] is not True or c['predicted_tokens_fed_back'] is not False or c['QD']!=24:raise ValueError('Wrong execution protocol')
        if r['method']=='leanstream':
            geometry=r['native_geometry']
            if not c['native_pre_enqueue'] or c['B_CTAs']!=32 or len(geometry)!=102 or {(g['layer'],g['stage']) for g in geometry}!={(l,s) for l in range(34) for s in range(3)} or any(g['grid']!=[610,32,128][g['stage']] or g['warps']!=4 for g in geometry):raise ValueError('Production native geometry mismatch')
        elif not r['io']['fixed_hot_immutable'] or r['io']['layer_decode_calls']!=34*count or r['io']['physical_bytes_read']<=0:raise ValueError('Missing genuine baseline layer/I/O execution')
    if tokenizer is not None:
        if tokenizer.encode(w['prompt_text'],add_special_tokens=True).ids!=prompt or tokenizer.encode(w['continuation_text'],add_special_tokens=False).ids!=tokens:raise ValueError('Input text/IDs do not match exact tokenizer')
        for i,prefix in enumerate(prefixes):
            if tokenizer.decode(tokens[:i],skip_special_tokens=True)!=prefix:raise ValueError('Displayed text is not the real fixed token prefix')
    return data

def render_fixed(data,output,preview=None,seconds=14,fps=12):
    from PIL import Image,ImageChops,ImageDraw,ImageFont
    fonts={n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',n) for n in [17,18,20,22,24,28]}
    runs=data['records'];w=data['workload'];duration=max(r['token_completion_ms'][-1] for r in runs);frames=[];palette=None;previous=None
    for frame in range(seconds*fps):
        elapsed=duration*min(frame/(seconds*fps-12),1)
        im=Image.new('RGB',(1600,700),'#0b1421');d=ImageDraw.Draw(im)
        d.text((25,16),'Fixed-token decode performance comparison',font=fonts[28],fill='#edf3fa')
        d.text((25,55),f'Ministral-3 8B | Jetson AGX Orin | {len(w["continuation_token_ids"])} fixed decode steps',font=fonts[20],fill='#acbcd0')
        d.text((1250,55),f'Decode: {elapsed/1000:5.1f} s',font=fonts[20],fill='#edf3fa')
        for i,line in enumerate(textwrap.wrap('Prompt: '+w['prompt_text'].replace('\n',' '),width=124)):d.text((25,96+24*i),line,font=fonts[20],fill='#d1dce9')
        d.text((25,150),'Same input sequence, independently measured runs',font=fonts[22],fill='#bdccde')
        for j,(r,name,color) in enumerate(zip(runs,['Block','One-shot','LeanStream'],['#b9c6d6','#e7c58c','#64d8ba'])):
            left=20+526*j;right=left+510;top=190;bottom=620
            d.rounded_rectangle((left,top,right,bottom),radius=8,fill='#101f30',outline='#334356',width=2)
            d.text((left+16,top+12),name,font=fonts[24],fill=color)
            d.text((left+16,top+46),f'{r["tokens_per_s"]:.3f} tokens/s | measured fixed decode',font=fonts[17],fill=color)
            count=bisect.bisect_right(r['token_completion_ms'],elapsed);text=w['decoded_prefixes'][count];lines=[]
            for paragraph in text.split('\n'):lines.extend(textwrap.wrap(paragraph,width=39,replace_whitespace=False,drop_whitespace=True) or [''])
            for i,line in enumerate(lines[-10:]):d.text((left+16,top+84+27*i),line,font=fonts[20],fill='#e6edf6')
            status='complete' if count==r['decode_steps'] else 'processing'
            d.text((left+16,bottom-29),f'{count}/{r["decode_steps"]} input tokens | {status}',font=fonts[17],fill=color)
        d.text((25,641),'Reference continuation fed as inputs; NOT independently generated answers.',font=fonts[20],fill='#bdccde')
        d.text((25,674),'Real forward completion timestamps | common replay scale | prefill/warmup/drain excluded',font=fonts[18],fill='#92a6c0')
        if palette is None:palette=im.quantize(colors=128)
        quantized=im.quantize(palette=palette,dither=0)
        if preview and frame==seconds*fps//2:quantized.convert('RGB').save(preview)
        if previous is None:encoded=quantized
        else:
            unchanged=ImageChops.difference(quantized,previous).point(lambda value:255 if value==0 else 0,mode='1')
            encoded=quantized.copy();encoded.paste(255,mask=unchanged)
        previous=quantized.copy();encoded.info['transparency']=255;frames.append(encoded)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0,optimize=False,disposal=1,transparency=255)


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,default=ROOT/'docs/media/fixed_token_decode_trace.json');p.add_argument('--output',type=Path,default=ROOT/'docs/media/speed_comparison.gif');p.add_argument('--preview',type=Path);p.add_argument('--diagnostic',action='store_true',help='Label the retained One-shot failure explicitly; no throughput comparison');p.add_argument('--tokenizer',type=Path,help='Optional exact tokenizer JSON to independently verify input IDs and every decoded prefix');a=p.parse_args()
    data=json.loads(a.trace.read_text());tokenizer=None
    if a.tokenizer:
        if hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()!=data['records'][0]['tokenizer_sha256']:raise ValueError('Tokenizer checksum mismatch')
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(a.tokenizer))
    validate_trace(data,tokenizer);a.output.parent.mkdir(parents=True,exist_ok=True);render(data,a.output,a.preview,diagnostic=a.diagnostic);print(a.output,a.output.stat().st_size,'bytes')
if __name__=='__main__':main()
