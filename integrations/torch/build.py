"""Build the optional bridge; no PyTorch headers or model assets are needed."""
import argparse
from pathlib import Path
import subprocess
import platform
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda-11.8'))
p.add_argument('--predictor',action='store_true',help='Also build the existing CPU predictor as a separate optional library')
p.add_argument('--testing',action='store_true',help='Enable a submission-failure injection entry point for optional tests only')
a=p.parse_args();root=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=True)
sources=['integrations/torch/native/bridge.cpp','backends/triton/enqueue.cpp','backends/triton/mapped.cpp','src/runtime/runtime.cpp','src/runtime/weight_manager.cpp','src/runtime/execution_queue.cpp','src/runtime/kernel_selector.cpp','src/io/stream_io.cpp']
subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-pthread',*(['-DLEANSTREAM_TORCH_TESTING','-DSTREAM_TESTING'] if a.testing else []),'-I'+str(root/'include'),'-I'+str(a.cuda/'include'),*map(lambda s:str(root/s),sources),'-L'+str(a.cuda/'lib64'),'-Wl,-rpath,'+str(a.cuda/'lib64'),'-lcudart','-lcuda','-o',str(a.output/'libleanstream_torch.so')],check=True)

if a.predictor:
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-fopenmp','-I'+str(root/'include'),*(['-march=armv8.2-a+fp16+dotprod'] if platform.machine()=='aarch64' else []),str(root/'predictor/predictor_cpu.cpp'),'-o',str(a.output/'libpredictor.so')],check=True)
