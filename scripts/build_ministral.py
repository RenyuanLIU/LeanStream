"""Build optional model-policy glue against the public runtime and Triton backend."""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--nvcc',default='/usr/local/cuda-11.8/bin/nvcc')
p.add_argument('--arch',default='sm_87')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
model=root/'examples/ministral'
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
native=model/'native'
sources=[native/x for x in ['leanstream_mapped.cu','leanstream_control.cpp','leanstream_pipeline.cpp',
    'ready_id_queue.cpp','stream_io_compat.cpp','predictor_shared_input.cpp','predictor_shared_input.cu']]
sources += [root/'src/runtime'/x for x in ['execution_queue.cpp','weight_manager.cpp','prediction_schedule.cpp','runtime.cpp','kernel_selector.cpp']]
sources += [root/'src/io/stream_io.cpp',root/'backends/triton/enqueue.cpp']
cmd=[a.nvcc,'-std=c++17','-O3','--fmad=false','-arch='+a.arch,'-Xcompiler=-fPIC',
    '-shared','-Xcompiler=-pthread','-DSTREAM_IO_V2','-I'+str(root/'include'),
    '-I'+str(root/'src/io'),*map(str,sources),'-ldl','-lcuda','-o',str(a.output/'libmapped.so')]
subprocess.run(cmd,check=True)
subprocess.run([a.nvcc,'-std=c++17','-O3','-shared','-Xcompiler=-fPIC',str(root/'backends/triton/mapped.cpp'),'-o',str(a.output/'libtriton_mapped.so')],check=True)
# Only the included predictor is needed; external implementations may replace this library.
import platform
subprocess.run(['g++','-std=c++17','-O3','-fopenmp','-fPIC','-shared',
    *(['-march=armv8.2-a+fp16+dotprod'] if platform.machine()=='aarch64' else []),
    '-I'+str(root/'include'),str(root/'predictor/predictor_cpu.cpp'),'-o',str(a.output/'libpredictor.so')],check=True)
print('Built public Runtime, production Triton enqueue, model adapter, mapped buffers and CPU predictor')
