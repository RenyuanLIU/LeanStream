"""Read-only structural discovery, never graph rewriting."""
from .contracts import ModelPoints

def resolve(root,path):
    value=root
    for part in path.split('.'):
        value=value[int(part)] if part.isdecimal() else getattr(value,part)
    return value

def inspect_model(model,points=None):
    points=points or ModelPoints.auto();candidates=[]
    paths=[points.layers] if points.layers!='auto' else ['model.layers','model.model.layers','layers']
    for path in paths:
        try:
            layers=resolve(model,path)
            if len(layers)>0:candidates.append((path,layers))
        except (AttributeError,TypeError,KeyError):pass
    report=dict(compatible=False,scope='structural inspection; arithmetic/readiness require user validation',ambiguities=[],missing=[],layers=[],points=None)
    if len(candidates)!=1:
        report['ambiguities']=[p for p,_ in candidates];report['missing']=['Specify a unique decoder layer path'];return report
    path,layers=candidates[0]
    chosen=ModelPoints(path,points.mlp,points.attention,points.pre_mlp_norm)
    seen=set()
    for i,layer in enumerate(layers):
        row=dict(index=i,path=f'{path}.{i}',compatible=False)
        try:
            mlp=resolve(layer,chosen.mlp);attention=resolve(layer,chosen.attention);norm=resolve(layer,chosen.pre_mlp_norm)
            if id(layer) in seen or id(mlp) in seen:raise ValueError('Shared/reentrant layers or MLPs unsupported')
            seen.update([id(layer),id(mlp)])
            g,u,d=mlp.gate_proj,mlp.up_proj,mlp.down_proj
            h,j=g.in_features,g.out_features
            if u.in_features!=h or u.out_features!=j or d.in_features!=j or d.out_features!=h:raise ValueError('Inconsistent gated projections')
            if any(getattr(p,'bias',None) is not None for p in (g,u,d)):raise ValueError('Bias unsupported by reference bundle')
            if norm.weight.numel()!=h or not callable(attention.forward):raise ValueError('Normalization/attention geometry')
            row.update(compatible=True,hidden=h,intermediate=j,mlp=f'{path}.{i}.{chosen.mlp}',attention=f'{path}.{i}.{chosen.attention}',norm=f'{path}.{i}.{chosen.pre_mlp_norm}')
        except (AttributeError,ValueError,TypeError) as e:row['reason']=str(e);report['missing'].append(f'{path}.{i}: {e}')
        report['layers'].append(row)
    report['compatible']=bool(report['layers']) and all(r['compatible'] for r in report['layers']);report['points']=chosen
    return report
