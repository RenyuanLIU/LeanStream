"""Reconstruct exact historical inputs locally from the original licensed CoQA dev data.

No dataset text, answers or prompt token IDs are included in release archives.
"""
import argparse,json,urllib.request
from pathlib import Path
if __package__:
    from .common import HERE,load,sha,write,record_lines
    from .assets import AssetError
else:
    from common import HERE,load,sha,write,record_lines
    from assets import AssetError

def reconstruct(dataset,assets):
    selection=load(HERE/'coqa_selection.json');dataset=Path(dataset);assets=Path(assets)
    if dataset.stat().st_size!=selection['dataset']['size'] or sha(dataset)!=selection['dataset']['sha256']:raise AssetError('Wrong CoQA dev-v1.0 file; refusing a different split/revision')
    raw=load(dataset)['data'];by_id={r['id']:r for r in raw}
    if len(raw)!=500 or len(by_id)!=500:raise AssetError('Expected complete original 500-conversation validation set')
    def columns(rows,keys):return {k:[r[k] for r in rows] for k in keys}
    answer_keys=['span_start','span_end','span_text','input_text','turn_id']
    docs=[]
    for i,item in enumerate(selection['records']):
        r=by_id[item['sample_id']]
        if item['index']!=i or r['source']!=item['source'] or len(r['questions'])!=item['turns']:raise AssetError('Fixed-500 source/order/turn mismatch')
        docs.append(dict(id=r['id'],source=r['source'],story=r['story'],questions=columns(r['questions'],['input_text','turn_id']),answers=columns(r['answers'],answer_keys),additional_answers={k:columns(r['additional_answers'][k],answer_keys) for k in ['0','1','2']}))
    if __package__:
        from .scoring.coqa import doc_to_text,doc_to_target
    else:
        from scoring.coqa import doc_to_text,doc_to_target
    from tokenizers import Tokenizer
    if not (assets/'metadata/tokenizer_compat.json').is_file():raise AssetError('Extract the exact runtime artifacts first: tokenizer compatibility metadata is missing')
    tokenizer=Tokenizer.from_file(str(assets/'metadata/tokenizer_compat.json'))
    prompts=[dict(index=i,sample_id=d['id'],input_ids=tokenizer.encode(doc_to_text(d),add_special_tokens=True).ids) for i,d in enumerate(docs)]
    references=[dict(index=i,sample_id=d['id'],answers=doc_to_target(d)) for i,d in enumerate(docs)]
    encoded=lambda rows:(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)).encode()
    contents={'benchmarks/coqa_docs.json':(json.dumps(docs,indent=2)+'\n').encode(),'benchmarks/coqa_fixed500_prompts.jsonl':encoded(prompts),'benchmarks/speed_prompts.jsonl':encoded(prompts),'benchmarks/coqa_fixed500_references.jsonl':encoded(references)}
    import hashlib
    for name,data in contents.items():
        item=selection['expected_files'][name]
        if len(data)!=item['size'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise AssetError('Reconstructed bytes differ from historical benchmark: '+name)
    # Validate everything before installing any output; never replace incompatible records.
    for name,data in contents.items():
        path=assets/name
        if path.exists() and path.read_bytes()!=data:raise AssetError('Incompatible existing benchmark: '+name)
    for name,data in contents.items():
        path=assets/name
        if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
    return dict(status='PASS',records=500,exact_historical_bytes=True,dataset_sha256=sha(dataset))

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--dataset',type=Path);p.add_argument('--download',action='store_true',help='Download original dev set under its original source terms');p.add_argument('--cache',type=Path);a=p.parse_args()
    if not a.dataset and not a.download:p.error('Supply --dataset or explicitly request --download after reviewing CoQA source terms')
    try:
        if not a.dataset:
            info=load(HERE/'coqa_selection.json')['dataset'];cache=a.cache or a.assets.parent/'.coqa-download';cache.mkdir(parents=True,exist_ok=True);a.dataset=cache/'coqa-dev-v1.0.json'
            if not a.dataset.exists():
                urls=[info['official_url'],f"https://huggingface.co/datasets/{info['mirror_repository']}/resolve/{info['mirror_revision']}/{info['mirror_file']}"]
                for url in urls:
                    try:
                        with urllib.request.urlopen(url,timeout=30) as r:data=r.read(info['size']+1)
                        import hashlib
                        if len(data)!=info['size'] or hashlib.sha256(data).hexdigest()!=info['sha256']:raise AssetError('Dataset download checksum mismatch')
                        temporary=a.dataset.with_suffix('.part');temporary.write_bytes(data);temporary.replace(a.dataset);break
                    except OSError:
                        if url==urls[-1]:raise
        print(json.dumps(reconstruct(a.dataset,a.assets)))
    except (OSError,ValueError,KeyError) as e:p.exit(1,str(e)+'\n')
if __name__=='__main__':main()
