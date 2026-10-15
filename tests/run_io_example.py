"""Run a synthetic-file example with isolated, automatically removed storage."""
from pathlib import Path
import subprocess
import sys
import tempfile
with tempfile.TemporaryDirectory(prefix='leanstream-example-',dir=Path.cwd()) as folder:
    weights=Path(folder)/'weights.bin'
    command=[sys.argv[1],str(weights)]
    if sys.argv[2]=='dynamic':
        weights.write_bytes(bytes(8*4096))
        command+=['8','4096']
    subprocess.run(command,check=True,timeout=35)
