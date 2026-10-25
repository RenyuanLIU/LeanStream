"""Inference contracts. No model download, native allocation, or compilation on import."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass(frozen=True)
class ModelPoints:
    layers: str = 'model.layers'
    mlp: str = 'mlp'
    attention: str = 'self_attn'
    pre_mlp_norm: str = 'post_attention_layernorm'

    @classmethod
    def auto(cls):
        return cls(layers='auto')

@dataclass(frozen=True)
class GroupLayout:
    logical_group: int
    physical_group: int
    file_offset: int
    bytes: int

@dataclass(frozen=True)
class WeightSpec:
    path: Path
    width: int
    layout: tuple
    scales: object  # CPU FP16 [physical_group, 2, 3, width/128]

    @property
    def group_count(self): return len(self.layout)
    @property
    def group_bytes(self): return 3*self.width  # two neurons, three signed INT4 rows

    def validate(self):
        n=self.group_count;b=self.group_bytes
        if n<3 or self.width<512 or self.width%512: raise ValueError('Group2 bundle requires width multiple of 512, >=512')
        if {x.logical_group for x in self.layout}!=set(range(n)) or {x.physical_group for x in self.layout}!=set(range(n)):
            raise ValueError('Logical and physical IDs must each be permutations')
        if any(x.bytes!=b or x.file_offset%512 or x.file_offset<0 for x in self.layout):raise ValueError('Invalid aligned extents')
        ranges=sorted((x.file_offset,x.file_offset+b) for x in self.layout)
        if any(a[1]>b_[0] for a,b_ in zip(ranges,ranges[1:])):raise ValueError('Overlapping weight extents')
        if not Path(self.path).is_file() or Path(self.path).stat().st_size<ranges[-1][1]:raise ValueError('Missing/truncated weight file')
        import torch
        if self.scales.device.type!='cpu' or self.scales.dtype!=torch.float16 or self.scales.numel()!=n*2*3*self.width//128 or not torch.isfinite(self.scales).all():
            raise ValueError('Expected finite CPU FP16 physical scale table')

@dataclass(frozen=True)
class CachePolicy:
    resident: tuple
    core: tuple

    def validate(self,n):
        for ids in (self.resident,self.core):
            if not ids or len(set(ids))!=len(ids) or any(type(i)!=int or not 0<=i<n for i in ids):raise ValueError('Invalid fixed resident/core IDs')
        if not set(self.core)<=set(self.resident) or len(self.core)%2:raise ValueError('Core must be an even-sized resident subset')
        if len(self.core)>=n:raise ValueError('At least one B slot required')

@dataclass(frozen=True)
class PredictionStage:
    predictor: Callable
    budget: int
    io_budget: int

@dataclass(frozen=True)
class PredictionSchedule:
    final: PredictionStage
    early: PredictionStage = None
    attention_mid: object = None

    def validate(self,n,mode):
        if mode not in ('basic','progressive'):raise ValueError('Mode must be basic or progressive')
        if self.attention_mid is not None:raise ValueError('Attention Mid is documented only in this integration')
        if (mode=='progressive')!=(self.early is not None):raise ValueError('Progressive requires early; Basic excludes early')
        for stage in (self.early,self.final):
            if stage and (not callable(stage.predictor) or not 0<stage.budget<=n or not 0<=stage.io_budget<=n):raise ValueError('Invalid prediction budget')

@dataclass(frozen=True)
class ComputeSpec:
    library: Path
    # Optional materialized module package: exposes runtime_split/persistent_split/reduce_split.
    bundle: str = None
    check_execution: bool = False  # correctness counters only, not a performance setting


def ranked_ids(value,n,budget):
    ids=list(value)
    if len(ids)>budget or any(type(i)!=int or not 0<=i<n for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Predictor must return unique, in-range Python integer logical IDs within budget')
    return ids
