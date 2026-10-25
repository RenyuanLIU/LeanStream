"""Fixed-token capture contracts; tests never run a model or download assets."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
from capture_fixed_tokens import prepare_input

class FixedTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('fixed_gif',ROOT/'tools/make_speed_gif.py');cls.gif=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.gif)

    def real_trace(self):
        path=ROOT/'docs/media/fixed_token_decode_trace.json'
        if not path.exists():self.skipTest('Measured fixed-token trace not yet captured')
        return json.loads(path.read_text())

    def test_real_fixed_trace(self):
        data=self.real_trace();self.gif.validate_trace(data)
        self.assertEqual(len(data['workload']['continuation_token_ids']),112)
        source=json.loads((ROOT/'docs/media/fixed_decode_input.json').read_text())
        for k in ['prompt_text','continuation_text']:self.assertEqual(source[k],data['workload'][k])
        self.assertTrue(all(r['decode_steps']==112 for r in data['records']))

    def test_reject_mismatched_workload_and_timing(self):
        data=self.real_trace()
        for change in ['input','timestamp','steps','rate','binding','feedback']:
            bad=copy.deepcopy(data);r=bad['records'][1]
            if change=='input':r['processed_token_ids'][0]+=1
            if change=='timestamp':r['token_completion_ms'][1]=r['token_completion_ms'][0]
            if change=='steps':r['decode_steps']-=1
            if change=='rate':r['tokens_per_s']*=2
            if change=='binding':bad['records'][2]['native_geometry'][1]['grid']=4
            if change=='feedback':r['configuration']['predicted_tokens_fed_back']=True
            with self.assertRaises(ValueError,msg=change):self.gif.validate_trace(bad)

    def test_input_validation_no_silent_truncation(self):
        class Encoded:
            def __init__(self,ids):self.ids=ids
        class Tokenizer:
            def __init__(self,count=112,eos=False):self.count=count;self.eos=eos
            def encode(self,text,add_special_tokens):return Encoded([1] if add_special_tokens else ([2] if self.eos else list(range(10,10+self.count))))
            def decode(self,ids,**kwargs):return 'reference' if len(ids)==self.count else ''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.json';path.write_text(json.dumps(dict(prompt_text='prompt',continuation_text='reference')))
            self.assertEqual(len(prepare_input(path,Tokenizer())['continuation_token_ids']),112)
            for tokenizer in [Tokenizer(99),Tokenizer(129),Tokenizer(eos=True)]:
                with self.assertRaises(ValueError):prepare_input(path,tokenizer)
            path.write_text(json.dumps(dict(prompt_text=' ',continuation_text='reference')))
            with self.assertRaises(ValueError):prepare_input(path,Tokenizer())

    def test_fixed_gif_deterministic(self):
        try:from PIL import Image
        except ImportError:self.skipTest('Optional Pillow missing')
        if not Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf').exists():self.skipTest('Optional font missing')
        data=self.real_trace()
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.gif';b=Path(directory)/'b.gif'
            self.gif.render(data,a,seconds=2,fps=8);self.gif.render(data,b,seconds=2,fps=8)
            self.assertEqual(a.read_bytes(),b.read_bytes())
            with Image.open(a) as image:self.assertEqual(image.size,(1600,700))

    def test_optional_cli(self):
        import os,subprocess
        for name in ['capture_fixed_tokens','fixed_token_worker']:
            result=subprocess.run([sys.executable,'-B','-m','examples.ministral.reproduce.'+name,'--help'],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
