"""Build original optional baseline code against the public CPU predictor."""
import argparse,platform,subprocess
from pathlib import Path
if __package__:
    from .common import PROJECT,HERE
else:
    from common import PROJECT,HERE
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda-11.8'));a=p.parse_args()
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
subprocess.run(['g++','-std=c++17','-O3',*(['-march=armv8.2-a+fp16+dotprod'] if platform.machine()=='aarch64' else []),'-fopenmp','-pthread','-fPIC','-shared','-I'+str(a.cuda/'include'),'-I'+str(PROJECT/'include'),str(HERE/'baselines/fixed_cache.cpp'),str(PROJECT/'predictor/predictor_cpu.cpp'),'-L'+str(a.cuda/'lib64'),'-lcudart','-o',str(a.output/'libfixed_cache.so')],check=True)
