"""Immutable artifact inventory and structural checks for the optional example."""
from pathlib import Path, PurePosixPath
import re
if __package__:
    from .common import load, rows, sha, VERSION, MODEL_REPO, MODEL_REVISION, HERE
else:
    from common import load, rows, sha, VERSION, MODEL_REPO, MODEL_REVISION, HERE

class AssetError(ValueError):
    pass

def safe_path(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in name:
        raise AssetError('Unsafe artifact path: ' + name)
    result = Path(root)/name
    # Local model symlinks are supported separately; downloaded bundle entries are regular files.
    if Path(root).resolve() not in result.resolve().parents:
        raise AssetError('Artifact escapes output directory: ' + name)
    return result

def descriptor(path):
    m = load(path)
    if m.get('version') != VERSION or m.get('model',{}).get('repo_id') != MODEL_REPO or m['model'].get('revision') != MODEL_REVISION:
        raise AssetError('Incompatible experiment/model identity')
    if not m.get('files'):
        raise AssetError('Artifact inventory is empty')
    for name, record in m['files'].items():
        safe_path(Path(path).parent, name)
        if not re.fullmatch('[0-9a-f]{64}',record['sha256']) or type(record['size']) is not int or record['size'] < 0:
            raise AssetError('Invalid checksum/size: ' + name)
    for name in m['model'].get('files',{}):
        if PurePosixPath(name).name!=name or name in {'.','..'} or '\\' in name:
            raise AssetError('Unsafe model path: '+name)
    return m

def group_mapping(groups, mapping):
    import numpy as np
    if groups.shape != (7168,2) or mapping.shape != (14336,) or not np.array_equal(np.sort(groups.ravel()),np.arange(14336)):
        raise AssetError('Invalid Group2 permutation')
    if not np.array_equal(mapping[groups.ravel()],np.arange(14336)):
        raise AssetError('Predictor/SSD group mapping mismatch')

def hot_ids(ids):
    if len(ids)!=2437 or len(set(ids))!=2437 or any(type(x)!=int or not 0<=x<14336 for x in ids):
        raise AssetError('Invalid immutable fixed-hot neuron IDs')

def require_layers(*records):
    if any(len(rows)!=34 for rows in records):raise AssetError('Incomplete layer set; expected 34')

def predictor_identity(fm,lm,layer,identity,parent_sha,source_sha):
    if fm['layer']!=layer or fm['padded_stride']!=1664 or fm['identity']['checkpoint']!=identity or lm['checkpoint']!=identity or lm['layer']!=layer or lm['padded_stride']!=512:
        raise AssetError('Predictor layer/geometry/checkpoint mismatch')
    if lm['parent_predictor_manifest_sha256']!=parent_sha or lm['source_sha256']!=fm['identity']['source_sha256'] or fm['identity']['source_sha256']!=source_sha:
        raise AssetError('Predictor parent/source identity mismatch')

def verify(root, full=True, model=True, model_root=None, scope="all"):
    import numpy as np
    root=Path(root).resolve();m=descriptor(root/'MANIFEST.json')
    released=load(HERE/'artifact_catalog.json')
    if m['files']!=released['files'] or m['model']!=released['model']:
        raise AssetError('Collection differs from the exact versioned release inventory; regenerated substitutes are not accepted')
    if scope not in ('all','runtime'):raise AssetError('Unknown verification scope')
    for name,record in m['files'].items():
        if scope=='runtime' and name.startswith('benchmarks/'):continue
        p=safe_path(root,name)
        if not p.is_file() or p.stat().st_size!=record['size'] or (full and sha(p)!=record['sha256']):
            raise AssetError('Missing or checksum/size mismatch: '+name)
    if model:
        for name,record in m['model']['files'].items():
            p=Path(model_root or root/'model')/name
            if not p.is_file() or p.stat().st_size!=record['size'] or (full and sha(p)!=record['sha256']):
                raise AssetError('Upstream model missing/incompatible: '+name)
    om=load(root/'oneshot_predictors/MANIFEST.json')
    if om.get('kind')!='experimental_pre_attention_ridge_W4A8' or om['model']['revision']!=MODEL_REVISION:
        raise AssetError('Wrong recalibrated One-shot predictor family')
    training=load(root/'oneshot_predictors/TRAINING_MANIFEST.json')
    for i in range(34):
        prefix=f'layer_{i:02d}/'
        for name in ['features_1638.bin','predictor_1638.bin','predictor_1638_scales.bin']:
            record=om['files'][prefix+name]
            if record!=training['files'][prefix+name] or m['files']['oneshot_predictors/'+prefix+name]!=record:
                raise AssetError('Recalibrated predictor original parameter identity mismatch')
        features=np.fromfile(root/'oneshot_predictors'/prefix/'features_1638.bin',np.int32)
        scales=np.fromfile(root/'oneshot_predictors'/prefix/'predictor_1638_scales.bin',np.float32)
        if features.shape!=(1638,) or len(np.unique(features))!=1638 or np.any(features<0) or np.any(features>=4096) or scales.shape!=(28672,) or not np.isfinite(scales).all() or np.any(scales<=0):
            raise AssetError('Invalid recalibrated predictor features/scales')
    wm=load(root/'leanstream_weights/g2_weights/manifest.json')
    if wm['geometry']!=dict(H=4096,I=14336,layers=34,groups=7168) or wm['identity']['format']!='leanstream_signed_int4_per128_g2_v0':
        raise AssetError('Unsupported packed Group2 format')
    identity=dict(checkpoint=MODEL_REPO,revision=MODEL_REVISION)
    if wm['identity']['checkpoint']!=identity: raise AssetError('Wrong packed checkpoint')
    rope=load(root/'metadata/rope.json');model_identity=load(root/'metadata/model_identity.json')
    if rope['checkpoint']!=identity or len(rope['inv_freq'])!=64 or not np.isfinite(rope['inv_freq']).all() or not np.isfinite(rope['factor']):raise AssetError('Wrong YaRN identity/frequencies')
    if model_identity['repo_id']!=MODEL_REPO or model_identity['revision']!=MODEL_REVISION or model_identity['config_sha256']!=m['model']['files']['config.json']['sha256']:raise AssetError('Model configuration identity mismatch')
    if model and rope['rope_parameters']!=load(Path(model_root or root/'model')/'config.json')['text_config']['rope_parameters']:raise AssetError('YaRN parameters differ from checkpoint')
    capture=load(root/'metadata/predictor_capture.json')
    if wm['identity']['capture_sha256']!=sha(root/'metadata/predictor_capture.json') or capture['identity']['checkpoint']!=identity:
        raise AssetError('Predictor capture identity mismatch')
    bm=load(root/'baseline_weights/storage_manifest.json');hot=load(root/'baseline_weights/fixed_hot_cache_coqa_17pct.json')
    if (bm['H'],bm['I'],bm['layers'],bm['record_bytes'],bm['physical_order'])!=(4096,14336,34,6144,'logical neuron ID'):
        raise AssetError('Unsupported individual-neuron baseline storage')
    if bm['fixed_hot_file_sha256']!=sha(root/'baseline_weights/fixed_hot_cache_coqa_17pct.json') or hot['model_identity']!=identity:
        raise AssetError('Fixed-hot model/storage identity mismatch')
    if bm['source_manifest']['identity']['checkpoint']!=identity or bm['group_size'] is not None:
        raise AssetError('Baseline checkpoint/grouping mismatch')
    canonical=dict(hot);digest=canonical.pop('artifact_sha256')
    import hashlib,json
    if hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=digest:
        raise AssetError('Fixed-hot canonical identity mismatch')
    require_layers(hot['per_layer'],wm['layers'],bm['records'])
    for i in range(34):
        name=f'layer_{i:02d}';fullp=root/'full_predictors'/name/'d1664';lite=root/'p1_lite_predictors'/name/'d512'
        fm=load(fullp/'manifest.json');lm=load(lite/'manifest.json')
        predictor_identity(fm,lm,i,identity,sha(fullp/'manifest.json'),capture['files'][name+'/source.npz'])
        for folder,meta in [(fullp,fm),(lite,lm)]:
            for f,h in meta['files'].items():
                if sha(folder/f)!=h:raise AssetError('Predictor checksum mismatch: '+name+'/'+f)
        with np.load(fullp/'runtime.npz') as data:groups=data['groups']
        meta=load(root/'leanstream_weights/g2_weights'/name/'metadata.json')
        group_mapping(groups,np.asarray(meta['logical_to_physical']))
        if (meta['bundle_size'],meta['quant_group_size'],meta['qvalue_convention'])!=(2,128,'signed_twos_complement_low_nibble_first_-7_to_7'):
            raise AssetError('Quantization layout mismatch')
        for f,h in wm['layers'][i]['files'].items():
            if m['files']['leanstream_weights/g2_weights/'+f]['sha256']!=h:raise AssetError('Packed manifest disagrees with inventory')
        if hot['per_layer'][i]['layer']!=i or bm['records'][i]['layer']!=i:
            raise AssetError('Baseline layer order mismatch')
        hot_ids(hot['per_layer'][i]['neuron_ids'])
        with np.load(fullp/'runtime.npz') as data:features=data['features']
        for dim,stride in [(512,512),(1638,1664)]:
            folder=root/'baseline_weights'/name
            values=np.fromfile(folder/f'features_{dim}.bin',np.int32)
            if not np.array_equal(values,features[:dim]) or len(np.unique(values))!=dim or np.any(values<0) or np.any(values>=4096):
                raise AssetError('Baseline predictor feature identity mismatch')
            if (folder/f'predictor_{dim}.bin').stat().st_size!=14336*stride or (folder/f'predictor_{dim}_scales.bin').stat().st_size!=14336*2*4:
                raise AssetError('Baseline predictor storage layout mismatch')
        for f,h in bm['records'][i]['files'].items():
            key='baseline_weights/'+name+'/'+f
            if m['files'][key]['sha256']!=h:raise AssetError('Baseline manifest disagrees with inventory')
        if (root/'baseline_weights'/name/'neurons.bin').stat().st_size!=14336*6144:raise AssetError('Wrong individual-neuron storage size')
    if scope=='runtime':return m
    prompts=rows(root/'benchmarks/coqa_fixed500_prompts.jsonl');docs=load(root/'benchmarks/coqa_docs.json');refs=rows(root/'benchmarks/coqa_fixed500_references.jsonl')
    if len(prompts)!=500 or len(docs)!=500 or len(refs)!=500 or any(p['index']!=i or p['sample_id']!=d['id'] or p['sample_id']!=r['sample_id'] for i,(p,d,r) in enumerate(zip(prompts,docs,refs))):
        raise AssetError('Fixed-500 record identity/order mismatch')
    if __package__:
        from .scoring.coqa import doc_to_target
    else:
        from scoring.coqa import doc_to_target
    if any(r['index']!=i or r['answers']!=doc_to_target(d) for i,(r,d) in enumerate(zip(refs,docs))):
        raise AssetError('Scorer-compatible references do not match frozen documents')
    if sha(root/'benchmarks/speed_prompts.jsonl')!=sha(root/'benchmarks/coqa_fixed500_prompts.jsonl'):
        raise AssetError('Speed workload is not the historical prompt file')
    return m
