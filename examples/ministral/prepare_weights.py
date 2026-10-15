"""Pack supplied canonical FP16 MLP rows in the validated Group2 INT4 format."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pack.pack_mistral import pack_layer
p=argparse.ArgumentParser()
p.add_argument('--canonical',type=Path,required=True,help='layer_XX.npz with gate/up/down FP16 arrays')
p.add_argument('--predictors',type=Path,required=True,help='Validated layer_XX/d1664/runtime.npz with groups')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for layer in range(34):
    name=f'layer_{layer:02d}'
    with np.load(a.canonical/(name+'.npz')) as data, np.load(a.predictors/name/'d1664/runtime.npz') as predictor:
        groups=predictor['groups']
        assert groups.shape==(7168,2)
        meta=pack_layer(a.output/name,data['gate'],data['up'],data['down'],bundle_size=2,physical_to_logical=groups.ravel())
    assert (meta['H'],meta['I'])==(4096,14336)
    files={str(Path(name)/f):hashlib.sha256((a.output/name/f).read_bytes()).hexdigest() for f in ['qweights.bin','scales.bin','metadata.json']}
    rows.append(dict(layer=layer,files=files))
(a.output/'manifest.json').write_text(json.dumps(dict(format='leanstream_signed_int4_per128_g2_v0',layers=rows),indent=2)+'\n')
