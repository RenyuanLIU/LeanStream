"""Configure and build the model-independent public components."""
import argparse
import subprocess
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('build'))
parser.add_argument('--cuda',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
subprocess.run(['cmake','-S',str(root),'-B',str(args.output),
                '-DLEANSTREAM_CUDA='+('ON' if args.cuda else 'OFF')],check=True)
subprocess.run(['cmake','--build',str(args.output),'-j','2'],check=True)
