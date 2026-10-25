"""Opt-in, inference-only PyTorch attachment. Import requires PyTorch, not model assets."""
from .contracts import ModelPoints,GroupLayout,WeightSpec,CachePolicy,PredictionStage,PredictionSchedule,ComputeSpec
from .inspection import inspect_model
from .attachment import attach_leanstream
__all__=['ModelPoints','GroupLayout','WeightSpec','CachePolicy','PredictionStage','PredictionSchedule','ComputeSpec','inspect_model','attach_leanstream']
from .predictor import NativeCPUPredictor
__all__.append('NativeCPUPredictor')
