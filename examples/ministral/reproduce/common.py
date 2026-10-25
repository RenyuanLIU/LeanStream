"""Optional experiment utilities; importing this module performs no downloads."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
PROJECT = EXAMPLE.parents[1]
MODEL_REPO = 'cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit'
MODEL_REVISION = 'e3a799844ba08f27fddbf118e154057eb0105889'
VERSION = 'leanstream-ministral3-8b-v1'
INDICES = [0, 2, 5, 6, 7, 8, 9, 11]

def bootstrap():
    sys.path[:0] = [str(EXAMPLE), str(PROJECT), str(HERE)]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def record_lines(path, values):
    with Path(path).open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

def environment(build):
    env = {k: v for k, v in os.environ.items() if not k.startswith('LEANSTREAM_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(PROJECT),
        LEANSTREAM_RELEASE_COMMIT=command_output(['git','-C',str(PROJECT),'rev-parse','HEAD']) or os.environ.get('LEANSTREAM_RELEASE_COMMIT','unavailable'), LEANSTREAM_PUBLIC_CONSUMER='progressive', LEANSTREAM_PUBLIC_WORKERS='32',
        LEANSTREAM_PUBLIC_READY_PAIRS='1', LEANSTREAM_PUBLIC_NATIVE_ENQUEUE='1',
        LEANSTREAM_PUBLIC_EXECUTION_CHECK='0', LEANSTREAM_STREAM_QD='24',
        LEANSTREAM_STREAM_PROFILE='0', LEANSTREAM_MAPPED_LIBRARY=str(Path(build).resolve()/'libtriton_mapped.so'))
    return env

def command_output(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 'unavailable: ' + str(e)

def provenance(assets, build, method):
    import importlib.metadata as metadata
    origins = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and (name.startswith('python.') or name.startswith('pack.') or name.startswith('scripts.') or name.startswith('backends.') or name.startswith('reproduce.') or name.startswith('baselines.') or name.startswith('scoring.') or name.startswith('examples.ministral.reproduce.')):
            path = Path(path).resolve()
            if PROJECT not in path.parents:
                raise RuntimeError('Unexpected runtime source: ' + name + ': ' + str(path))
            origins[name] = str(path)
    libraries = {}
    maps = Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            if any(n in line for n in ['libmapped.so', 'libpredictor.so', 'libfixed_cache.so', 'libtriton_mapped.so']):
                path = Path(line.split()[-1]).resolve()
                if path.parent != Path(build).resolve():
                    raise RuntimeError('Unexpected native library: ' + str(path))
                libraries[str(path)] = sha(path)
    clocks = {}
    for pattern in ['/sys/devices/system/cpu/cpu0/cpufreq/scaling_*freq',
        '/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/*freq',
        '/sys/kernel/debug/bpmp/debug/clk/emc/rate']:
        import glob
        for name in glob.glob(pattern):
            try: clocks[name] = Path(name).read_text().strip()
            except OSError: clocks[name] = 'not readable'
    packages = {}
    for name in ['torch', 'triton', 'transformers', 'numpy', 'safetensors', 'tokenizers', 'huggingface-hub']:
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = 'not installed'
    manifest = load(Path(assets)/'MANIFEST.json')
    import torch
    device=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else dict(available=False)
    return dict(cuda_device=device,public_commit=command_output(['git','-C',str(PROJECT),'rev-parse','HEAD']) or os.environ.get('LEANSTREAM_RELEASE_COMMIT','unavailable'),
        source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in PROJECT.rglob('*')
            if p.is_file() and '.git' not in p.parts and p.suffix in {'.py','.cpp','.cu','.cuh','.h'}},
        modules=origins, native_libraries=libraries, method=method, model=manifest['model'],
        artifact_manifest_sha256=sha(Path(assets)/'MANIFEST.json'), artifact_files=manifest['files'],
        hardware=platform.uname()._asdict(), software=packages, power_mode=command_output(['nvpmodel','-q']),
        clocks=clocks, emc_observability='unavailable unless recorded above',
        storage=command_output(['findmnt','-T',str(Path(assets).resolve())]),
        qd=24, attention='unsplit Direct-GQA', kernel_selection='validated 32-CTA fallback; no example LUT')
