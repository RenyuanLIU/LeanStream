"""Optional packaging/download/trace tests; fixtures are not measurement evidence."""
import copy,hashlib,importlib.util,io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from common import sha
from assets import AssetError,require_layers,predictor_identity
from archives import extract_archive,release_manifest
from download_assets import http_download,SafeRedirect

class Release(unittest.TestCase):
    def test_layers_and_predictor_identity(self):
        require_layers(list(range(34)))
        with self.assertRaises(AssetError):require_layers(list(range(33)))
        identity=dict(checkpoint='expected',revision='pinned')
        fm=dict(layer=0,padded_stride=1664,identity=dict(checkpoint=identity,source_sha256='source'))
        lm=dict(layer=0,padded_stride=512,checkpoint=identity,parent_predictor_manifest_sha256='parent',source_sha256='source')
        predictor_identity(fm,lm,0,identity,'parent','source')
        for key,value in [('layer',1),('padded_stride',1024),('parent_predictor_manifest_sha256','wrong')]:
            bad=dict(lm,**{key:value})
            with self.assertRaises(AssetError):predictor_identity(fm,bad,0,identity,'parent','source')

    def archive(self,root,entries):
        path=root/'test.tar.gz'
        with tarfile.open(path,'w:gz') as tar:
            for name,data,kind in entries:
                info=tarfile.TarInfo(name);info.size=len(data)
                if kind=='link':info.type=tarfile.SYMTYPE;info.linkname='/outside';info.size=0
                tar.addfile(info,io.BytesIO(data))
        return path,dict(name=path.name,size=path.stat().st_size,sha256=sha(path),files=['layer_00/data'])

    def test_extraction_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=b'exact artifact';path,record=self.archive(root,[('layer_00/data',data,'file')]);catalog=dict(files={'layer_00/data':dict(size=len(data),sha256=hashlib.sha256(data).hexdigest())});out=root/'out'
            extract_archive(path,record,out,catalog);target=out/'layer_00/data';mtime=target.stat().st_mtime_ns
            extract_archive(path,record,out,catalog);self.assertEqual(target.stat().st_mtime_ns,mtime)
            path.write_bytes(path.read_bytes()+b'bad')
            with self.assertRaises(AssetError):extract_archive(path,record,root/'corrupt',catalog)

    def test_unsafe_and_duplicate_members(self):
        for entries in [[('../outside',b'x','file')],[('layer_00/data',b'x','link')],[('layer_00/data',b'x','file'),('layer_00/data',b'x','file')]]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);path,record=self.archive(root,entries);catalog=dict(files={'layer_00/data':dict(size=1,sha256=hashlib.sha256(b'x').hexdigest())})
                with self.assertRaises(AssetError):extract_archive(path,record,root/'out',catalog)
                self.assertFalse((root.parent/'outside').exists())

    def test_missing_and_changed_payload(self):
        for entries in [[],[('layer_00/data',b'wrong','file')]]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);path,record=self.archive(root,entries);catalog=dict(files={'layer_00/data':dict(size=5,sha256=hashlib.sha256(b'exact').hexdigest())})
                with self.assertRaises(AssetError):extract_archive(path,record,root/'out',catalog)

    def test_http_range_resume(self):
        class Response(io.BytesIO):
            headers={'Content-Range':'bytes 3-5/6'}
            def getcode(self):return 206
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'archive';target.with_name('archive.part').write_bytes(b'abc')
            record=dict(size=6,sha256=hashlib.sha256(b'abcdef').hexdigest())
            with patch('download_assets.OPENER.open',return_value=Response(b'def')) as opened:
                http_download('https://example.org/archive',target,record)
                self.assertEqual(opened.call_args[0][0].get_header('Range'),'bytes=3-')
            self.assertEqual(target.read_bytes(),b'abcdef')
            with patch('download_assets.OPENER.open') as opened:http_download('https://example.org/archive',target,record);opened.assert_not_called()

    def test_redirect_does_not_leak_auth(self):
        import urllib.request
        request=urllib.request.Request('https://api.github.com/assets/1',headers={'Authorization':'Bearer test-only-fixture'})
        redirected=SafeRedirect().redirect_request(request,None,302,'',{},'https://release-assets.githubusercontent.com/file')
        self.assertIsNone(redirected.get_header('Authorization'))

    def test_versioned_manifest_completeness(self):
        manifest=HERE/'asset_manifest.json';m=release_manifest(manifest)
        self.assertEqual(len(m['archives']),9)
        self.assertTrue(all(r['size']<int(1.8*1024**3) for r in m['archives']))
        for mutate in ['missing','duplicate','checkpoint']:
            bad=copy.deepcopy(m)
            if mutate=='missing':bad['archives'][0]['files'].pop()
            if mutate=='duplicate':bad['archives'][1]['name']=bad['archives'][0]['name']
            if mutate=='checkpoint':bad['checkpoint']['revision']='main'
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'manifest.json';path.write_text(json.dumps(bad))
                with self.assertRaises(AssetError):release_manifest(path)

    def test_module_cli_arguments(self):
        import os,subprocess
        for name in ['download_assets','download_model','verify_assets','prepare_coqa','reproduce_speed','reproduce_accuracy','smoke','capture_tokens','capture_worker']:
            result=subprocess.run([sys.executable,'-B','-m','examples.ministral.reproduce.'+name,'--help'],cwd=HERE.parents[2],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,(name,result.stderr))
            self.assertIn('usage:',result.stdout)

    def test_prompt_file_preserves_default_and_exact_text(self):
        from capture_tokens import read_prompt,PROMPT
        self.assertEqual(read_prompt(),PROMPT)
        original=json.loads((HERE.parents[2]/"docs/media/diagnostics/original_token_generation_trace.json").read_text())
        self.assertEqual(read_prompt(),original["prompt"]["prompt_text"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prompt.txt'
            text='A public story.\n\nQuestion: Why?\nAnswer: '
            path.write_text(text,encoding='utf-8')
            self.assertEqual(read_prompt(path),text)
            path.write_text(' \n')
            with self.assertRaises(ValueError):read_prompt(path)
            path.write_bytes(b'\xff')
            with self.assertRaises(UnicodeError):read_prompt(path)
            with self.assertRaises(OSError):read_prompt(path.parent/'missing.txt')

    def test_one_shot_numerical_evidence_identity(self):
        root=HERE.parents[2]/'docs/media/diagnostics'
        proof=json.loads((root/'one_shot_numerical_check.json').read_text())
        original=json.loads((root/'original_token_generation_trace.json').read_text())
        self.assertEqual(proof['original_prompt_trace_sha256'],sha(root/'original_token_generation_trace.json'))
        self.assertEqual(proof['baseline_source_sha256'],sha(HERE/'baselines/fixed_cache.py'))
        self.assertTrue(proof['all_finite'] and proof['first_24_ids_match_original'])
        self.assertEqual(proof['original_prompt_layer_checks'],34*(proof['warmup_decode_calls']+proof['checked_decode_calls']))
        record=original['records'][1]
        self.assertEqual(proof['generated_token_ids_including_prefill'],[record['prefill_token_id']]+record['generated_token_ids'][:23])
        self.assertFalse(proof['timing_used_in_gif'])

    def test_retained_candidate_trace_alignment(self):
        root=HERE.parents[2]/'docs/media/diagnostics'
        plan=json.loads((root/'prompt_plan.json').read_text())
        self.assertEqual(plan['order'],['garden','library'])
        self.assertEqual((root.parent/'token_generation_trace.json').read_bytes(),(root/'library_token_generation_trace.json').read_bytes())
        spec=importlib.util.spec_from_file_location('candidate_gif',HERE.parents[2]/'tools/make_speed_gif.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for name in plan['order']:
            data=json.loads((root/(name+'_token_generation_trace.json')).read_text())
            mod.validate_trace(data)
            self.assertEqual(data['prompt']['prompt_text'],(root/(name+'_prompt.txt')).read_text())
            for record in data['records']:
                self.assertEqual(record['max_new_tokens'],plan['max_new_tokens'])
                self.assertTrue(record['configuration']['greedy'])
                if record['method']=='leanstream':
                    self.assertEqual(len(record['native_geometry']),102)
                    self.assertTrue(all(row['grid']==[610,32,128][row['stage']] and row['warps']==4 for row in record['native_geometry']))
                else:
                    self.assertTrue(record['io']['fixed_hot_immutable'])
                    self.assertGreater(record['io']['physical_bytes_read'],0)

    def test_readme_links(self):
        import re
        root=HERE.parents[2]
        for name in ['README.md','examples/ministral/README.md','docs/reproduce_ministral3_8b.md','docs/media/README.md','docs/media/diagnostics/README.md']:
            path=root/name
            for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                target=link.split('#')[0]
                if target and '://' not in target:self.assertTrue((path.parent/target).exists(),(name,target))

    def test_gif_deterministic(self):
        try:from PIL import Image
        except ImportError:self.skipTest('Optional Pillow unavailable')
        if not Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf').is_file():self.skipTest('Optional DejaVu font unavailable')
        spec=importlib.util.spec_from_file_location('speed_gif',HERE.parents[2]/'tools/make_speed_gif.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        path=HERE.parents[2]/'docs/media/token_generation_trace.json'
        if not path.is_file():self.skipTest('Real capture not installed yet')
        data=json.loads(path.read_text());mod.validate_trace(data)
        with tempfile.TemporaryDirectory() as d:
            for diagnostic in [False,True]:
                a=Path(d)/'a.gif';b=Path(d)/'b.gif';mod.render(data,a,seconds=2,fps=8,diagnostic=diagnostic);mod.render(data,b,seconds=2,fps=8,diagnostic=diagnostic)
                self.assertEqual(sha(a),sha(b))
                with Image.open(a) as image:self.assertEqual(image.size,(1600,620))

    def test_real_trace_schema_and_tokenizer_check(self):
        spec=importlib.util.spec_from_file_location('speed_gif',HERE.parents[2]/'tools/make_speed_gif.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        path=HERE.parents[2]/'docs/media/token_generation_trace.json'
        if not path.is_file():self.skipTest('Real capture not installed yet')
        data=json.loads(path.read_text());mod.validate_trace(data)
        bad=copy.deepcopy(data);bad['records'][0]['token_completion_ms'][1]=bad['records'][0]['token_completion_ms'][0]
        with self.assertRaises(ValueError):mod.validate_trace(bad)
        bad=copy.deepcopy(data);bad['records'][1]['prompt_input_ids_sha256']='0'*64
        with self.assertRaises(ValueError):mod.validate_trace(bad)
        class WrongDecoder:
            def decode(self,*a,**k):return 'fixture intentionally disagrees'
        with self.assertRaises(ValueError):mod.validate_trace(data,WrongDecoder())

if __name__=='__main__':unittest.main()
