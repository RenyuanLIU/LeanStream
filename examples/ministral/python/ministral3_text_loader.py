"""Text-only Ministral3 checkpoint adapter for the existing Mistral runtime.

The source remains the model-specific compressed INT4 checkpoint. This adapter
does not instantiate or patch the vision tower. It is intentionally separate
from the Falcon GPTQ loader because the checkpoint formats differ.
"""
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import MistralConfig, MistralForCausalLM


def unpack_weight(packed, scale, shape):
    if packed.dtype != torch.int32 or scale.ndim != 2:
        raise ValueError('Expected symmetric packed INT4 and per-group scales')
    rows, cols = map(int, shape)
    if tuple(packed.shape) != (rows, (cols + 7) // 8) or scale.shape[0] != rows or cols % scale.shape[1]:
        raise ValueError('Compressed INT4 shape mismatch')
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    q = ((packed.unsqueeze(-1) >> shifts) & 15).reshape(rows, -1)[:, :cols].to(torch.int8) - 8
    group_size = cols // scale.shape[1]
    return q.to(scale.dtype) * scale.repeat_interleave(group_size, dim=1)


class PackedLinear(nn.Module):
    """Checkpoint's packed MLP projection; dense prefill remains INT4-backed."""
    bits = 4
    def __init__(self, packed, scale, shape):
        super().__init__()
        self.out_features, self.in_features = map(int, shape)
        self.register_buffer('weight_packed', packed)
        self.register_buffer('weight_scale', scale)

    def forward(self, x):
        weight = unpack_weight(self.weight_packed, self.weight_scale,
                               (self.out_features, self.in_features)).to(x.dtype)
        return F.linear(x, weight)


class FixedYarnRope(nn.Module):
    """Checkpoint YaRN frequencies for the bounded CoQA context, with growth."""
    def __init__(self, inv_freq, factor):
        super().__init__()
        self.register_buffer('inv_freq', inv_freq.float(), persistent=False)
        self.factor = float(factor)
        self.max_seq_len_cached = 0

    def forward(self, x, seq_len=None):
        if seq_len is None:
            raise ValueError('RoPE sequence length required')
        if torch.is_tensor(seq_len):
            freqs = seq_len.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * self.inv_freq.to(x.device)
            emb = torch.cat((freqs, freqs), dim=-1)
            return (emb.cos() * self.factor).to(x.dtype), (emb.sin() * self.factor).to(x.dtype)
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            emb = torch.cat((torch.outer(t, self.inv_freq.to(x.device)),) * 2, dim=-1)
            self.cos_cached = emb.cos() * self.factor
            self.sin_cached = emb.sin() * self.factor
            self.max_seq_len_cached = seq_len
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


def load_text(checkpoint_path, rope_path, device='cuda'):
    if '2.1.0a0' in torch.__version__:
        # Jetson vendor PyTorch exposes working SDPA despite prerelease
        # ordering below Transformers' 2.1.1 version check.
        import transformers.modeling_utils as modeling_utils
        modeling_utils.is_torch_sdpa_available = lambda: True
    root = Path(checkpoint_path)
    raw = json.loads((root / 'config.json').read_text())
    text = raw['text_config']
    if (text['hidden_size'], text['intermediate_size'], text['num_hidden_layers']) != (4096, 14336, 34):
        raise ValueError('Ministral3 text geometry mismatch')
    rope = json.loads(Path(rope_path).read_text())
    if rope['rope_parameters'] != text['rope_parameters']:
        raise ValueError('YaRN parameters differ from checkpoint')
    if raw['quantization_config']['quant_method'] != 'compressed-tensors':
        raise ValueError('Unexpected source INT4 format')
    config = MistralConfig(hidden_size=4096, intermediate_size=14336,
                           num_hidden_layers=34, num_attention_heads=text['num_attention_heads'],
                           num_key_value_heads=text['num_key_value_heads'],
                           hidden_act=text['hidden_act'], rms_norm_eps=text['rms_norm_eps'],
                           vocab_size=text['vocab_size'], sliding_window=None,
                           max_position_embeddings=8192, rope_theta=text['rope_parameters']['rope_theta'])
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = MistralForCausalLM(config)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    with ExitStack() as stack:
        shards = {name: stack.enter_context(safe_open(root / name, framework='pt', device='cpu'))
                  for name in set(index.values())}
        def get(key):
            return shards[index[key]].get_tensor(key)
        def quant(prefix):
            packed = get(prefix + '.weight_packed').to(device)
            scale = get(prefix + '.weight_scale').to(device)
            shape = get(prefix + '.weight_shape').tolist()
            return packed, scale, shape
        for layer_id, layer in enumerate(model.model.layers):
            layer.self_attn.rotary_emb = FixedYarnRope(torch.tensor(rope['inv_freq']), rope['factor'])
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                prefix = f'language_model.model.layers.{layer_id}.mlp.{name}'
                setattr(layer.mlp, name, PackedLinear(*quant(prefix)))
        for name, parameter in list(model.named_parameters()):
            owner_name, leaf = name.rsplit('.', 1)
            owner = model.get_submodule(owner_name)
            key = 'language_model.' + name
            if '.self_attn.' in name and leaf == 'weight':
                weight = unpack_weight(*quant(key[:-7]))
            else:
                weight = get(key)
            if tuple(weight.shape) != tuple(parameter.shape):
                raise ValueError(f'Checkpoint shape mismatch: {name}')
            setattr(owner, leaf, nn.Parameter(weight.to(device=device, dtype=torch.float16),
                                              requires_grad=False))
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Uninitialized text parameter')
    return model.eval()
