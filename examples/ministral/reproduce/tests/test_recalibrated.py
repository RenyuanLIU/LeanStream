"""Default One-shot contract tests; no GPU, model or downloads required."""
import ast,hashlib,json,tempfile,types,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
class Recalibrated(unittest.TestCase):
 def test_distinct_complete_namespace(self):
  c=json.loads((HERE/'artifact_catalog.json').read_text());files=c['files']
  for i in range(34):
   for name in ['features_1638.bin','predictor_1638.bin','predictor_1638_scales.bin']:
    self.assertIn(f'oneshot_predictors/layer_{i:02d}/'+name,files)
  self.assertNotEqual(files['oneshot_predictors/layer_00/predictor_1638.bin']['sha256'],files['baseline_weights/layer_00/predictor_1638.bin']['sha256'])
 def install_function(self,calls):
  tree=ast.parse((HERE/'baselines/one_shot.py').read_text());function=next(x for x in tree.body if isinstance(x,ast.FunctionDef));ns=dict(Path=Path,json=json,hashlib=hashlib,__file__=str(HERE/'baselines/one_shot.py'),Runtime=lambda *a,**kw:calls.append((a,kw)))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'install','exec'),ns);return ns['install']
 def test_missing_parameters_fail_no_512_fallback(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'v2'):self.install_function([])(object(),Path(d)/'baseline_weights','native')
 def test_early_proxy_exactly_once(self):
  tree=ast.parse((HERE/'baselines/recalibrated.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MLP');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='begin');ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'begin','exec'),ns)
  calls=[];r=types.SimpleNamespace(decode=True,mode='one_shot',early_domain='mlp_norm_proxy',start=lambda *a:calls.append(a));obj=types.SimpleNamespace(runtime=r,index=7,early_norm=lambda x:x+10)
  ns['begin'](obj,None,[3]);self.assertEqual(calls,[(7,13)])
  r.decode=False;ns['begin'](obj,None,[3]);self.assertEqual(len(calls),1)
 def test_default_install_is_not_historical_runtime(self):
  text=(HERE/'baselines/one_shot.py').read_text();self.assertIn('from .recalibrated import Runtime',text);self.assertIn('features=1638',text);self.assertIn("early_domain='mlp_norm_proxy'",text);self.assertIn("/'oneshot_predictors'",text)
 def test_nine_archives_cover_new_parameters(self):
  m=json.loads((HERE/'asset_manifest.json').read_text());self.assertEqual(len(m['archives']),9)
  a=next(a for a in m['archives'] if 'oneshot-predictors' in a['name']);self.assertIn('oneshot_predictors/MANIFEST.json',a['files']);self.assertTrue(all(x.startswith('oneshot_predictors/') for x in a['files']))
if __name__=='__main__':unittest.main()
