"""Default One-shot: recalibrated1638-feature early normalized residual, one prefetch."""
from pathlib import Path
import hashlib,json
from .recalibrated import Runtime

def install(model,assets,native,sanity=False):
    predictors=Path(assets).parent/'oneshot_predictors'
    catalog=json.loads((Path(__file__).parents[1]/'artifact_catalog.json').read_text())
    manifest=predictors/'MANIFEST.json'
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=catalog['files']['oneshot_predictors/MANIFEST.json']['sha256']:
        raise ValueError('Missing/incompatible recalibrated One-shot predictors; download the complete v2 asset collection')
    return Runtime(model,assets,native,'one_shot',sanity=sanity,features=1638,
                   early_domain='mlp_norm_proxy',predictor_root=predictors)
