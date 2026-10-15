"""Export the included checkpoint loader's canonical FP16 MLP rows offline."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from python.ministral3_text_loader import load_text,unpack_weight
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--rope',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
model=load_text(a.model,a.rope,device='cpu')
for i,layer in enumerate(model.model.layers):
    values={}
    for source,target in [('gate_proj','gate'),('up_proj','up'),('down_proj','down')]:
        proj=getattr(layer.mlp,source)
        values[target]=unpack_weight(proj.weight_packed,proj.weight_scale,(proj.out_features,proj.in_features)).numpy().astype(np.float16)
    np.savez(a.output/f'layer_{i:02d}.npz',**values)
