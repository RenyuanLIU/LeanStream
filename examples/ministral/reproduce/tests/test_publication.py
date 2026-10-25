"""Publication gates use synthetic metadata, never fabricated release evidence."""
import copy,hashlib,io,json,os,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import check_publication as check

SHA='a'*40
class Publication(unittest.TestCase):
    def metadata_api(self,private=False,draft=False,annotated=False,duplicates=False):
        def api(url):
            if '/git/ref/' in url:return dict(object=dict(type='tag' if annotated else 'commit',sha=SHA))
            if '/git/tags/' in url:return dict(object=dict(type='commit',sha=SHA))
            if '/releases/' in url:
                asset=dict(name='test.tar.gz',state='uploaded',size=1)
                return dict(id=1,tag_name='version',draft=draft,published_at=None if draft else '2026-10-24T00:00:00Z',assets=[asset,asset] if duplicates else [asset])
            return dict(private=private,visibility='private' if private else 'public')
        return api

    def test_public_release_gates(self):
        for annotated in [False,True]:
            commit,release,assets=check.public_metadata('owner/repo','version',self.metadata_api(annotated=annotated));self.assertEqual(commit,SHA)
        for kwargs in [dict(private=True),dict(draft=True),dict(duplicates=True)]:
            with self.assertRaises(ValueError):check.public_metadata('owner/repo','version',self.metadata_api(**kwargs))

    def test_no_credentials_in_subprocess_environment(self):
        original=dict(GH_TOKEN='test-only-fixture',GITHUB_TOKEN='test-only-fixture',HF_TOKEN='test-only-fixture',HF_HUB_OFFLINE='1',PYTHONPATH='unrelated-source')
        with patch.dict(os.environ,original):env=check.anonymous_environment(Path('/public-source'),Path('/output'))
        for key in ['GH_TOKEN','GITHUB_TOKEN','HF_TOKEN','HF_HUB_OFFLINE']:self.assertNotIn(key,env)
        self.assertEqual(env['HF_HUB_DISABLE_IMPLICIT_TOKEN'],'1');self.assertEqual(env['PYTHONPATH'],'/public-source');self.assertEqual(env['HF_HOME'],'/output/hf-cache')

    def test_http_requests_are_anonymous(self):
        with patch.dict(os.environ,dict(GH_TOKEN='test-only-fixture',GITHUB_TOKEN='test-only-fixture')):
            with patch.object(check.OPENER,'open',return_value=io.BytesIO(b'{"public": true}')) as opened:
                self.assertEqual(check.request_json('https://api.github.com/repos/owner/repo'),{'public':True})
                self.assertIsNone(opened.call_args[0][0].get_header('Authorization'))

    def test_complete_source_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'source.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                for name in check.REQUIRED:
                    member=tarfile.TarInfo('root/'+name);member.size=7;tar.addfile(member,io.BytesIO(b'fixture'))
            check.extract_source(archive,root/'source')
            self.assertTrue(all((root/'source'/name).read_bytes()==b'fixture' for name in check.REQUIRED))

    def test_manifest_approval_and_attached_sizes(self):
        import archives
        m=json.loads((HERE/'asset_manifest.json').read_text());catalog=(HERE/'artifact_catalog.json').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);d=root/'examples/ministral/reproduce';d.mkdir(parents=True);(d/'artifact_catalog.json').write_bytes(catalog)
            def install(data):
                raw=(json.dumps(data,indent=2)+'\n').encode();(d/'asset_manifest.json').write_bytes(raw)
                return {**{r['name']:dict(size=r['size']) for r in data['archives']},'asset_manifest.json':dict(size=len(raw)),'SHA256SUMS':dict(size=1)}
            pending=copy.deepcopy(m);pending['redistribution']='PENDING_OWNER_APPROVAL';assets=install(pending)
            with self.assertRaises(ValueError):check.approved_manifest(root,archives.PUBLIC_REPOSITORY,archives.ASSET_TAG,assets)
            approved=copy.deepcopy(m);approved['publication']['status']='PUBLISHED';approved['redistribution']='APPROVED';assets=install(approved)
            _,expected=check.approved_manifest(root,archives.PUBLIC_REPOSITORY,archives.ASSET_TAG,assets);self.assertEqual(len(expected),10)
            assets[approved['archives'][0]['name']]['size']+=1
            with self.assertRaises(ValueError):check.approved_manifest(root,archives.PUBLIC_REPOSITORY,archives.ASSET_TAG,assets)

    def test_checksums_cannot_omit_or_change_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'archive').write_bytes(b'exact');digest=hashlib.sha256(b'exact').hexdigest();expected={'archive':dict(size=5,sha256=digest)};sums=root/'SHA256SUMS';sums.write_text(digest+'  archive\n');check.verify_sums(sums,root,expected)
            for text in ['',digest+'  ../escape\n',digest+'  archive\n'+digest+'  archive\n','0'*64+'  archive\n']:
                sums.write_text(text)
                with self.assertRaises(ValueError):check.verify_sums(sums,root,expected)
            sums.write_text(digest+'  archive\n');(root/'archive').write_bytes(b'wrong')
            with self.assertRaises(ValueError):check.verify_sums(sums,root,expected)

    def test_source_archive_rejects_links_and_traversal(self):
        for name,kind in [('root/../escape','file'),('root/link','link'),('other-root/file','file')]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);archive=root/'source.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    info=tarfile.TarInfo('root');info.type=tarfile.DIRTYPE;tar.addfile(info)
                    member=tarfile.TarInfo(name);member.size=1
                    if kind=='link':member.type=tarfile.SYMTYPE;member.linkname='/outside';member.size=0
                    tar.addfile(member,io.BytesIO(b'x'))
                with self.assertRaises(ValueError):check.extract_source(archive,root/'source')
                self.assertFalse((root/'escape').exists())

    def test_supported_source_must_preserve_inference(self):
        manifest=dict(source_base_revision=SHA)
        def api(url):
            if '/compare/' in url:return dict(status='ahead')
            if '/commits/' in url:return dict(tree=dict(sha=SHA))
            return dict(truncated=False,tree=[dict(type='blob',path='src/runtime/runtime.cpp',sha='blob')])
        check.verify_source_basis('owner/repo','b'*40,manifest,api)
        calls=[]
        def changed(url):
            result=api(url)
            if '/trees/' in url:
                calls.append(url)
                if len(calls)>1:result['tree'][0]['sha']='changed'
            return result
        with self.assertRaises(ValueError):check.verify_source_basis('owner/repo','b'*40,manifest,changed)

    def test_cli_help_is_optional(self):
        import subprocess
        result=subprocess.run([sys.executable,'-B','-m','examples.ministral.reproduce.check_publication','--help'],cwd=HERE.parents[2],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--release-tag',result.stdout)

if __name__=='__main__':unittest.main()
