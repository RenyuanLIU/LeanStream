"""Check emitted protocol against the public implementation; run numerical.py separately."""
import sys
sys.dont_write_bytecode=True
from materialize import validate
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args();validate(a.bundle);print('PASS: resident/persistent/reduction protocol bodies match public source')
