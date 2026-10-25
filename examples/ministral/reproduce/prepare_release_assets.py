"""Author-only byte-preserving staging. No training, weight packing or uploads."""
import argparse,json,shutil,hashlib
from pathlib import Path
if __package__:
    from .common import VERSION,MODEL_REPO,MODEL_REVISION,sha,write,record_lines,INDICES,HERE
else:
    from common import VERSION,MODEL_REPO,MODEL_REVISION,sha,write,record_lines,INDICES,HERE

def main():
    p=argparse.ArgumentParser()
    for name in ['leanstream-weights','full-predictors','p1-lite-predictors','baseline-artifacts','predictor-capture','rope','model','prompts','coqa-docs','tokenizer-compat','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--catalog-output',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def copy(source,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if sha(source)!=sha(target):raise RuntimeError('Staging checksum mismatch: '+name)
    wm=json.loads((a.leanstream_weights/'g2_weights/manifest.json').read_text())
    bm=json.loads((a.baseline_artifacts/'storage_manifest.json').read_text())
    copy(a.leanstream_weights/'g2_weights/manifest.json','leanstream_weights/g2_weights/manifest.json')
    for layer in wm['layers']:
        for name,h in layer['files'].items():
            assert sha(a.leanstream_weights/'g2_weights'/name)==h
            copy(a.leanstream_weights/'g2_weights'/name,'leanstream_weights/g2_weights/'+name)
    for i in range(34):
        name=f'layer_{i:02d}'
        for source,prefix,dim,names in [(a.full_predictors,'full_predictors','d1664',['runtime.npz','w4.bin','scales4.bin','manifest.json','speculative_hot.npy','speculative_hot.json']),(a.p1_lite_predictors,'p1_lite_predictors','d512',['runtime.npz','w4.bin','scales4.bin','manifest.json'])]:
            for f in names:copy(source/name/dim/f,prefix+'/'+name+'/'+dim+'/'+f)
        for f,h in bm['records'][i]['files'].items():
            assert sha(a.baseline_artifacts/name/f)==h
            copy(a.baseline_artifacts/name/f,'baseline_weights/'+name+'/'+f)
    for name in ['storage_manifest.json','fixed_hot_cache_coqa_17pct.json']:copy(a.baseline_artifacts/name,'baseline_weights/'+name)
    for source,name in [(a.predictor_capture,'metadata/predictor_capture.json'),(a.rope,'metadata/rope.json'),(a.tokenizer_compat,'metadata/tokenizer_compat.json'),(a.prompts,'benchmarks/coqa_fixed500_prompts.jsonl'),(a.prompts,'benchmarks/speed_prompts.jsonl'),(a.coqa_docs,'benchmarks/coqa_docs.json')]:copy(source,name)
    if __package__:
        from .scoring.coqa import doc_to_target
    else:
        from scoring.coqa import doc_to_target
    docs=json.loads(a.coqa_docs.read_text())
    record_lines(out/'benchmarks/coqa_fixed500_references.jsonl',[dict(index=i,sample_id=d['id'],answers=doc_to_target(d)) for i,d in enumerate(docs)])
    write(out/'metadata/model_identity.json',dict(repo_id=MODEL_REPO,revision=MODEL_REVISION,config_sha256=sha(a.model/'config.json')))
    write(out/'metadata/protocol.json',dict(seed=42,max_new_tokens=64,eos_id=2,greedy=True,indices=INDICES,reverse_indices=list(reversed(INDICES)),speed_decode_calls=16,warmup_prompts=1,accuracy_warmup_decode_calls=4,speed_stop='EOS before each decode call',accuracy_stop='EOS/64 generation; decoded newline Q: offline',dataset='CoQA validation; frozen ordered 500; 100 each mctest/race/cnn/wikipedia/gutenberg',upstream_dataset_revision=None,upstream_revision_status='not recorded by historical export; exact frozen content SHA256 is authoritative',historical_coqa_docs_sha256=sha(a.coqa_docs),historical_prompt_sha256=sha(a.prompts)))
    (out/'README.md').write_text('# LeanStream Ministral-3 8B v1 artifacts\n\nByte-preserving copies of validated experiment binaries. No preprocessing is required by the inference recipe. Model checkpoint files are downloaded separately at the pinned revision. Public hosting is not yet configured. Redistribution approval is pending; do not upload this staging collection.\n')
    (out/'LICENSE_OR_NOTICE').write_text('UNRELEASED — redistribution review required.\nModel-derived files: checkpoint declares Apache-2.0; preserve upstream notices and identify the requantized/reordered derivative. Predictor/hot-set artifacts derive from calibrated model activations. CoQA prompts/references retain mixed upstream source terms: CC BY-SA 4.0, MSR-LA, RACE terms, CNN Apache terms. Token IDs are not a license exemption. No blanket MIT license is applied to data. Owner approval and applicable notices/permissions are required before public hosting. See docs/reproduce_ministral3_8b.md.\n')
    copy(HERE/'notices/Apache-2.0.txt','LICENSES/Apache-2.0.txt')
    files={str(f.relative_to(out)):dict(size=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file()}
    model_files={f.name:dict(size=f.stat().st_size,sha256=sha(f)) for f in sorted(a.model.glob('*')) if f.is_file()}
    manifest=dict(version=VERSION,status='UNPUBLISHED',redistribution='PENDING_REVIEW',hosting=dict(repo_id=None,revision=None),supported_methods=['block','one-shot','leanstream','accuracy'],model=dict(repo_id=MODEL_REPO,revision=MODEL_REVISION,files=model_files),files=files,total_bytes=sum(f['size'] for f in files.values()),model_download_bytes=sum(f['size'] for f in model_files.values()))
    write(out/'MANIFEST.json',manifest)
    (out/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {n}\n" for n,r in files.items())+sha(out/'MANIFEST.json')+'  MANIFEST.json\n')
    if a.catalog_output:write(a.catalog_output,manifest)
    print(json.dumps(dict(files=len(files),bytes=manifest['total_bytes'],model_bytes=manifest['model_download_bytes'],status='UNPUBLISHED')))
if __name__=='__main__':main()
