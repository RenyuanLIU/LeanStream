"""Small optional-example contracts; no checkpoints, network, or GPU required."""
import ast,ctypes as C,hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from assets import AssetError,hot_ids,group_mapping,safe_path,descriptor
from common import VERSION,MODEL_REPO,MODEL_REVISION
from download_assets import transfer

class Contracts(unittest.TestCase):
    def test_comparison_requires_both_directions_and_pools_time(self):
        from compare_results import compare
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def run(direction,n,total):
                path=root/('block-'+direction);path.mkdir()
                (path/'summary.json').write_text(json.dumps(dict(status='PASS',method='block',direction=direction,decode_tokens=n,total_decode_ms=total,sanity=False)))
            run('forward',2,10)
            with self.assertRaisesRegex(ValueError,'Incomplete'):compare(root)
            run('reverse',1,20);result=compare(root)[0]
            self.assertEqual(result['ms_per_token'],10)
            self.assertEqual(result['tokens_per_s'],100)

    def test_resume_validation_rejects_corrupt_or_duplicate_rows(self):
        from worker import validate_token_records
        prompts=[dict(sample_id='a'),dict(sample_id='b')]
        good=dict(index=0,sample_id='a',identity='key',generated_ids=[3,2])
        validate_token_records([good],prompts,'key',100,2)
        for change in [dict(index=1),dict(identity='other'),dict(generated_ids=[]),dict(generated_ids=[3]),dict(generated_ids=[2,3,2]),dict(generated_ids=[-1,2]),dict(generated_ids=[101,2])]:
            with self.assertRaises(ValueError):validate_token_records([{**good,**change}],prompts,'key',100,2)
        with self.assertRaises(ValueError):validate_token_records([good,good],prompts,'key',100,2)

    def test_fixed_hot(self):
        hot_ids(list(range(2437)))
        for ids in [list(range(2436)),[0]*2437,list(range(2436))+[14336]]:
            with self.assertRaises(AssetError):hot_ids(ids)
    def test_group_mapping(self):
        import numpy as np
        groups=np.arange(14336).reshape(7168,2);group_mapping(groups,np.arange(14336))
        for mapping in [np.arange(14336)[::-1],np.zeros(14336,dtype=int)]:
            with self.assertRaises(AssetError):group_mapping(groups,mapping)
    def test_local_download_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.write_bytes(b'weights are unchanged')
            record=dict(size=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest());target=root/'out'
            transfer(source,target,record);mtime=target.stat().st_mtime_ns;transfer(source,target,record)
            self.assertEqual(target.stat().st_mtime_ns,mtime)
            target.write_bytes(b'corruption')
            with self.assertRaises(AssetError):transfer(source,target,record)
            with self.assertRaises(AssetError):transfer(source,root/'bad',dict(size=1,sha256='0'*64))
    def test_downloader_local_cli(self):
        from unittest.mock import patch
        import download_assets
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();model=root/'model';model.mkdir()
            (source/'data').write_bytes(b'exact');(model/'config.json').write_bytes(b'{}')
            item=lambda p:dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            catalog=root/'catalog.json';catalog.write_text(json.dumps(dict(version=VERSION,redistribution='PENDING_REVIEW',hosting=dict(repo_id=None,revision=None),model=dict(repo_id=MODEL_REPO,revision=MODEL_REVISION,files={'config.json':item(model/'config.json')}),files={'data':item(source/'data')})))
            out=root/'out'
            args=['download_assets.py','--descriptor',str(catalog),'--output',str(out),'--local-source',str(source),'--model-source',str(model)]
            with patch.object(sys,'argv',args),patch('assets.verify') as verification:
                download_assets.main();verification.assert_called_once_with(out)
            self.assertEqual((out/'data').read_bytes(),b'exact')
            self.assertEqual((out/'model/config.json').read_bytes(),b'{}')

    def test_paths_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ['../outside','/absolute','x/../../outside','x\\outside']:
                with self.assertRaises(AssetError):safe_path(d,name)
            p=Path(d)/'MANIFEST.json';p.write_text(json.dumps(dict(version=VERSION,model=dict(repo_id=MODEL_REPO,revision=MODEL_REVISION),files={'data':dict(size=1,sha256='0'*64)})))
            self.assertEqual(descriptor(p)['version'],VERSION)
            m=json.loads(p.read_text());m['model']['revision']='main';p.write_text(json.dumps(m))
            with self.assertRaises(AssetError):descriptor(p)
    def test_real_baseline_scheduling_methods(self):
        # Execute the original Python methods with non-GPU collaborators to assert the submission order.
        tree=ast.parse((HERE/'baselines/fixed_cache.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MLP')
        events=[]
        class Kernel:
            def __getitem__(self,grid):return lambda *args,**kwargs:events.append('compute')
        env=dict(nn=types.SimpleNamespace(Module=object),C=C,PROGRAMS=717,H=4096,sparse=Kernel(),reduce=Kernel())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'baseline-methods','exec'),env)
        cls=env['MLP']
        for mode in ['blocking','one_shot']:
            events.clear();r=types.SimpleNamespace(decode=True,mode=mode,collect=False,sanity=False,ctx=1)
            r.start=lambda index,x:events.append(('predict_and_read',x))
            r.check=lambda status:None
            r.lib=types.SimpleNamespace(fc_wait=lambda ctx:events.append('required_reads_ready'),fc_counts=lambda *args:None)
            m=object.__new__(cls);m.runtime=r;m.index=0
            for name in ['cache','staging','scales','desc','partial','y']:setattr(m,name,types.SimpleNamespace(tensor=None))
            m.begin(None,['early']);m.forward('current')
            expected='current' if mode=='blocking' else 'early'
            self.assertEqual(events,[('predict_and_read',expected),'required_reads_ready','compute','compute'])
    def test_scorer_known_multireference(self):
        from scoring.coqa import compute_scores,process_results
        self.assertEqual(compute_scores(['The cat.'],'cat'),dict(em=1.,f1=1.))
        self.assertEqual(compute_scores(['yes','no'],'yes'),dict(em=.5,f1=.5))
        self.assertEqual(compute_scores(['a b c'],'b c'),dict(em=1.,f1=1.))
    def test_provenance_rejects_hidden_runtime(self):
        from common import provenance
        fake=types.ModuleType('python.hidden_private');fake.__file__='/outside/runtime.py';sys.modules[fake.__name__]=fake
        try:
            with self.assertRaisesRegex(RuntimeError,'Unexpected runtime source'):provenance('.', '.', 'block')
        finally:del sys.modules[fake.__name__]
if __name__=='__main__':unittest.main()
