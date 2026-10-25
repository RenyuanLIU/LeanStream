"""Optional CPU control/contract tests. Native GPU evidence is run.py --mode validate."""
import dataclasses
import importlib.util
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch
import sys
import os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1) # Explicit test process policy matches the reference OpenMP predictor.
from leanstream.integrations.torch import *
from leanstream.integrations.torch.contracts import ranked_ids
from examples.custom_pytorch.run import ToyModel,pack,predictor

class FakeNative:
    # Control-unit test only, never advertised as real I/O/GPU evidence.
    def __init__(self):self.active=False;self.closed=False
    def abort(self):self.active=False
    def drain(self):pass
    def close(self):self.closed=True
class FakeBundle:
    def __init__(self,w,p,c):self.weights=w;self.policy=p;self.compute=c;self.native=FakeNative()
    def close(self):self.native.close()

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.spec,_=pack(Path(self.tmp.name));self.model=ToyModel().eval();self.policy=CachePolicy((0,1,2),(0,1))
        self.schedule=PredictionSchedule(final=PredictionStage(predictor,5,5),early=PredictionStage(predictor,4,4))
    def test_inspection_read_only(self):
        before=[(n,id(m)) for n,m in self.model.named_modules()];r=inspect_model(self.model)
        self.assertTrue(r['compatible']);self.assertEqual([(n,id(m)) for n,m in self.model.named_modules()],before)
        self.assertEqual(r['layers'][0]['hidden'],512)
    def test_manual_and_ambiguous_paths(self):
        self.assertTrue(inspect_model(self.model,ModelPoints('layers'))['compatible'])
        self.model.model=torch.nn.Module();self.model.model.layers=torch.nn.ModuleList([self.model.layers[0]])
        self.assertFalse(inspect_model(self.model)['compatible']);self.assertTrue(inspect_model(self.model)['ambiguities'])
        self.assertFalse(inspect_model(self.model,ModelPoints('unknown'))['compatible'])
    def test_shared_layers_rejected(self):
        self.model.layers.append(self.model.layers[0])
        self.assertFalse(inspect_model(self.model,ModelPoints('layers'))['compatible'])

    def test_bad_geometry_and_bias(self):
        self.model.layers[0].mlp.up_proj=torch.nn.Linear(511,16,bias=False)
        self.assertFalse(inspect_model(self.model)['compatible'])
    def test_replacement_hooks_prefill_detach(self):
        original=self.model.layers[0].mlp
        # This unit tests forwarding/routing, not GEMM math. Use an original-MLP
        # spy and pointwise attention; real arithmetic is checked on the GPU.
        original.forward=lambda x:x*.125
        self.model.layers[0].self_attn=torch.nn.Identity()
        attention=self.model.layers[0].self_attn;x=torch.randn(1,3,512)
        with torch.inference_mode():expected=self.model(x)
        with patch('leanstream.integrations.torch.attachment.Group2Bundle',FakeBundle):
            h=attach_leanstream(self.model,points=ModelPoints('layers'),weights=[self.spec],compute=ComputeSpec(Path('unused')),cache_policy=[self.policy],prediction=self.schedule)
            self.assertIsNot(self.model.layers[0].mlp,original);self.assertIs(self.model.layers[0].self_attn,attention)
            self.assertEqual(len(self.model.layers[0]._forward_pre_hooks),1)
            with torch.inference_mode():actual=self.model(x)
            torch.testing.assert_close(actual,expected);self.assertEqual(actual.dtype,expected.dtype)
            with self.assertRaises(ValueError):attach_leanstream(self.model,points=ModelPoints('layers'),weights=[],compute=None,cache_policy=[],prediction=self.schedule)
            h.close();h.close();self.assertIs(self.model.layers[0].mlp,original);self.assertFalse(self.model.layers[0]._forward_hooks)
    def test_alignment_permutations_overlaps(self):
        self.spec.validate()
        for changes in [dict(layout=self.spec.layout[:-1]+(self.spec.layout[0],)),dict(layout=(dataclasses.replace(self.spec.layout[0],file_offset=1),)+self.spec.layout[1:]),dict(width=256),dict(path=Path(self.tmp.name)/'missing')]:
            with self.assertRaises(ValueError):dataclasses.replace(self.spec,**changes).validate()
    def test_predictor_ids_and_budgets(self):
        self.assertEqual(ranked_ids([7,0],8,2),[7,0])
        for ids in [[1,1],[-1],[8],[0,1,2]]:
            with self.assertRaises(ValueError):ranked_ids(ids,8,2)
        for stage in [PredictionStage(predictor,0,1),PredictionStage(predictor,2,9)]:
            with self.assertRaises(ValueError):PredictionSchedule(final=stage).validate(8,'basic')
    def test_modes_and_mid(self):
        self.schedule.validate(8,'progressive')
        with self.assertRaises(ValueError):self.schedule.validate(8,'basic')
        with self.assertRaises(ValueError):dataclasses.replace(self.schedule,attention_mid=True).validate(8,'progressive')
    def test_cache_core_contract(self):
        for policy in [CachePolicy((0,0),(0,)),CachePolicy((0,1),(2,3)),CachePolicy((0,1),(0,))]:
            with self.assertRaises(ValueError):policy.validate(8)
    def test_materialized_protocol_and_editable_math(self):
        spec=importlib.util.spec_from_file_location('materializer',ROOT/'templates/triton/materialize.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        target=Path(self.tmp.name)/'custom';m.materialize(target,2);m.validate(target)
        self.assertIn('* 2.0',(target/'group_arithmetic.py').read_text())
        p=target/'persistent.py';p.write_text(p.read_text().replace('head>=total','head>total'))
        with self.assertRaises(ValueError):m.validate(target)
    @unittest.skipUnless(os.environ.get('LEANSTREAM_TEST_PREDICTOR'),'Supply optional CPU predictor library for ABI smoke')
    def test_cpu_predictor_abi(self):
        import numpy as np
        weights=np.stack([np.full((2,32),.01*(i+1),dtype=np.float32) for i in range(4)])
        predictor=NativeCPUPredictor(os.environ['LEANSTREAM_TEST_PREDICTOR'],weights=weights,scales=np.ones((4,2),np.float32),rows=4,stride=32,budget=2,precision=32,input_provider=lambda *args:(np.ones(32,np.float32),1.0))
        try:self.assertEqual(predictor('mlp_input',0,None,2),[3,2])
        finally:predictor.close()

    def test_unready_future_never_waits(self):
        from leanstream.integrations.torch.attachment import SparseMLP
        r=SparseMLP(self.model.layers[0].mlp,None,0,FakeBundle(self.spec,self.policy,None))
        self.assertIs(r.publish(Future(),self.schedule.final,'mlp_input'),False)

if __name__=='__main__':unittest.main()
