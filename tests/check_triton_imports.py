"""Check that the optional model adapter has no missing project dependencies."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
example=root/'examples/ministral'
for path in [*example.rglob('*.py'),*(root/'backends/triton').glob('*.py')]:
    tree=ast.parse(path.read_text(),filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module:
            package=node.module.split('.')[0]
            if package in {'python','scripts','kernels','pack','backends'}:
                base=root if package=='backends' else example
                candidate=base.joinpath(*node.module.split('.'))
                assert candidate.with_suffix('.py').exists() or (candidate/'__init__.py').exists(),(path,node.module)
print('PASS: production backend and optional model adapter project imports are self-contained')
