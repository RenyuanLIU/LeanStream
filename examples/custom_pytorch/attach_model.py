"""Explicit local-HF-model attachment recipe; artifacts/predictor supplied by user.

Not a validated Qwen E2E example. --inspect-only does not modify the loaded model.
"""
import argparse,importlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from leanstream.integrations.torch import *
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--inspect-only',action='store_true');p.add_argument('--layers',default='model.layers');p.add_argument('--manifest',type=Path);p.add_argument('--predictor',help='module:factory, returns (early_predictor, final_predictor)');p.add_argument('--library',type=Path);p.add_argument('--bundle');p.add_argument('--input-ids',type=Path,help='JSON list of token IDs, length>=2');a=p.parse_args()
from transformers import AutoModelForCausalLM
model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
points=ModelPoints(a.layers);print(inspect_model(model,points))
if a.inspect_only:raise SystemExit(0)
if not all([a.manifest,a.predictor,a.library,a.input_ids]):p.error('Supply manifest, predictor factory, library and input-ids')
meta=json.loads(a.manifest.read_text());base=a.manifest.resolve().parent
weights=[];policies=[]
for row in meta['layers']:
    weights.append(WeightSpec(base/row['file'],row['width'],tuple(GroupLayout(**x) for x in row['layout']),torch.from_numpy(np.load(base/row['scales'],allow_pickle=False))))
    policies.append(CachePolicy(tuple(row['resident']),tuple(row['core'])))
module,name=a.predictor.split(':');early,final=getattr(importlib.import_module(module),name)(meta)
schedule=PredictionSchedule(early=PredictionStage(early,meta['early_budget'],meta['early_io_budget']),final=PredictionStage(final,meta['final_budget'],meta['final_io_budget']))
ids=json.loads(a.input_ids.read_text())
if len(ids)<2:p.error('Use >=2 prompt tokens; single-token prefill requires explicit dense-prefill management')
with torch.inference_mode(),attach_leanstream(model,points=points,weights=weights,compute=ComputeSpec(a.library,a.bundle),cache_policy=policies,prediction=schedule) as handle:
    result=model.generate(torch.tensor([ids],device='cuda'),max_new_tokens=4,do_sample=False)
    handle.drain();print(result.tolist())
