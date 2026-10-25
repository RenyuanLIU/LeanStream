"""Emit editable arithmetic with byte-preserved public READY/claim kernel bodies."""
import argparse
import ast
import textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def function(path,name):
    text=path.read_text();node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name==name)
    start=min([node.lineno]+[d.lineno for d in node.decorator_list]);return '\n'.join(text.splitlines()[start-1:node.end_lineno])+'\n'

def resident_parts(base):
    source=function(base/'core.py','runtime_split');tree=ast.parse(source)
    node=next(n for n in tree.body[0].body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id=='DYNAMIC' for x in ast.walk(n.test)))
    lines=source.splitlines();body='\n'.join(lines[node.body[0].lineno-1:node.body[-1].end_lineno])
    helper='@triton.jit\ndef _resident_arithmetic(program,p,limit,X,Cache,Staging,Scales,Desc,H,Partial,Stamp,WIDTH:tl.constexpr,G:tl.constexpr):\n'+textwrap.indent(textwrap.dedent(body),'    ')+'\n'
    call='        _resident_arithmetic(program,p,limit,X,Cache,Staging,Scales,Desc,H,Partial,Stamp,WIDTH,G)'
    # Only arithmetic is factored out. Epoch wait, inactive zeroing and finish stay framework-owned.
    source=source.replace(body,call).replace('    if not DYNAMIC or program*4<limit:', '    lane=tl.arange(0,256)\n    if not DYNAMIC or program*4<limit:')
    return source,helper

def materialize(output,multiplier=1):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);base=ROOT/'backends/triton'
    imports='import triton\nimport triton.language as tl\nfrom backends.triton.primitives import _finish, _clock\n'
    weight=function(base/'core.py','weight')
    # A concrete static specialization example. Edit this generated arithmetic module.
    weight=weight.replace('return (q.to(tl.float32)*scale).to(tl.float16).to(tl.float32)',f'return (q.to(tl.float32)*scale).to(tl.float16).to(tl.float32) * {float(multiplier)!r}')
    resident,helper=resident_parts(base)
    (output/'group_arithmetic.py').write_text(imports+weight+'\n'+function(base/'persistent.py','_compute_pair')+'\n'+helper)
    (output/'resident.py').write_text(imports+'from .group_arithmetic import _resident_arithmetic\n'+resident)
    (output/'persistent.py').write_text(imports+'from .group_arithmetic import _compute_pair\n'+function(base/'persistent.py','persistent_split'))
    (output/'reduction.py').write_text(imports+function(base/'reduction.py','reduce_split'))
    (output/'__init__.py').write_text('from .resident import runtime_split\nfrom .persistent import persistent_split\nfrom .reduction import reduce_split\n')
    validate(output)

def validate(output):
    base=ROOT/'backends/triton'
    for target,source,name in [('resident.py','core.py','runtime_split'),('persistent.py','persistent.py','persistent_split'),('reduction.py','reduction.py','reduce_split')]:
        expected=resident_parts(base)[0] if target=='resident.py' else function(base/source,name)
        if function(Path(output)/target,name)!=expected:raise ValueError('Framework protocol/body changed: '+target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--weight-multiplier',type=float,choices=[1,2],default=1);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:validate(a.output)
    else:materialize(a.output,a.weight_multiplier)
