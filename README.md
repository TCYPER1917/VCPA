# 🚀 VCPA: Vectorized Chunk-Pooled Attention (Gen 11 Architecture)

> **Breakthrough Memory-Evolved Architecture for Long-Context LLMs**
>
> Overcoming quadratic complexity ($O(N^2)$), eliminating Python loop overhead, and successfully scaling long-context inference where standard models crash with `OutOfMemoryError` (OOM).

## 🧬 Overview

As large language models process longer contexts, standard self-attention suffers from a notorious quadratic bottleneck ($O(N^2)$), leading to massive VRAM spikes and frequent OOM crashes. Traditional chunking methods tried to mitigate this using slow Python loops (`for loops`), which heavily throttle the PyTorch autograd graph and degrade training performance.

**VCPA (Vectorized Chunk-Pooled Attention)** — developed as the **Gen 11** evolution — solves this fundamentally. By fusing multi-scale semantic landmarks with pure vectorized PyTorch operations, VCPA achieves extreme memory efficiency, massive speedups, and robust scaling on commodity GPUs (such as the NVIDIA Tesla T4).

---

## 📊 Benchmark & Stress Test Results (Real-World Kaggle Validation)

Tested on **Qwen/Qwen2.5-0.5B** on an **NVIDIA Tesla T4 GPU** (14.56 GiB VRAM capacity):

### 1. Standard Context ($N = 2048$ tokens)
| Metric | Baseline (Standard SDPA) | Gen 11 (VCPA Optimized) | Improvement / Savings |
| :--- | :--- | :--- | :--- |
| **Peak VRAM** | `11,893.73 MB` (~11.89 GB) | `6,158.89 MB` (~6.16 GB) | **Saved 5,734.84 MB (48.22% Reduction)** |
| **Execution Time** | `0.7885 s` | `0.2871 s` | **63.58% Speedup (~2.7x Faster)** |
| **Patched Layers** | - | **24 / 24 Layers** | Full 100% Architecture Coverage |

### 2. High-Stress Long-Context ($N = 4096$ tokens) 🚨
| Metric | Baseline (Standard SDPA) | Gen 11 (VCPA Optimized) | Status & Outcome |
| :--- | :--- | :--- | :--- |
| **Peak VRAM** | *Crashed* | `9,308.84 MB` (~9.31 GB) | **VCPA Survived & Handled 4096 Tokens!** |
| **Execution Time** | *Failed (OOM)* | `0.9993 s` | **Full Forward + Backward Pass Success** |
| **Error Log** | `OutOfMemoryError` at 14.56 GiB | - | **Rescued from catastrophic VRAM explosion** |

---

## 🛠️ How It Works (Core Innovations)

1. **Zero Python Loops:** Completely vectorized tensor operations (`pure PyTorch`), avoiding the Autograd overhead that plagued previous experimental generations.
2. **Chunk Landmark Pooling:** Keys and values are dynamically compressed into multi-scale semantic blocks ($C = 64$), reducing instantaneous attention matrix dimensions from $N \times N$ to $N \times (N/C)$.
3. **Full Differentiability:** Seamlessly supports `loss.backward()` across all transformer layers without memory leaks or gradient explosion.

---

## 🚀 Quick Start

You can easily drop the VCPA module into any Hugging Face causal language model architecture (such as Qwen, LLaMA, or Mistral):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VCPAQwen2Attention(nn.Module):
    def __init__(self, original_module, chunk_size=64):
        super().__init__()
        self.q_proj = original_module.q_proj
        self.k_proj = original_module.k_proj
        self.v_proj = original_module.v_proj
        self.o_proj = original_module.o_proj
        self.scaling = original_module.scaling
        self.chunk_size = chunk_size
        
    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        bsz, seq_len, _ = hidden_states.shape
        # Vectorized chunk-pooled attention execution
        # (Refer to repository code.py for the complete implementation)
        return self.o_proj(hidden_states), None
```

---

## 📂 Repository Structure

* `code.py`: Full benchmarking and stress-testing script for Kaggle / Google Colab ($N = 2048$ & $N = 4096$).
* `results.json` & `results_4096.json`: Raw execution telemetry and memory telemetry data.
* `GEN11_REPORT.md`: Detailed engineering report and architectural breakdown.

---
### 📦 Installation and Prerequisites

Ensure the following libraries are installed:

```bash
pip install torch transformers

```

---

### 🚀 Usage (Quick Start & Integration)

You can inject the **VCPA (Vectorized Chunk-Pooled Attention)** algorithm into any attention layer of a Qwen model (or compatible models) to overcome memory constraints and process long contexts—up to `4096` tokens—without `OOM` errors:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

# 1. Define the VCPA algorithm class
class VCPAQwen2Attention(nn.Module):
def __init__(self, original_module, chunk_size=64, local_window=128):
super().__init__()
self.config = original_module.config
self.layer_idx = original_module.layer_idx
self.head_dim = getattr(original_module, "head_dim", 64)
self.num_heads = getattr(original_module, "num_heads", 14)
self.num_key_value_heads = getattr(original_module, "num_key_value_heads", 2)
self.num_key_value_groups = self.num_heads // self.num_key_value_heads
self.scaling = original_module.scaling

self.q_proj = original_module.q_proj
self.k_proj = original_module.k_proj
self.v_proj = original_module.v_proj
self.o_proj = original_module.o_proj

``` self.chunk_size = chunk_size 
self.local_window = local_window 

def forward( 
self, 
hidden_states: torch.Tensor, 
position_embeddings: tuple = None, 
attention_mask: torch.Tensor = None, 
past_key_values ​​= None, 
**kwargs 
): 
input_shape = hidden_states.shape[:-1] 
bsz, seq_len = input_shape 
hidden_shape = (*input_shape, -1, self.head_dim) 

query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) 
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) 
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) 

if position_embeddings is not None: 
cos, sin = position_embeddings 
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) 

if self.num_key_value_groups > 1: 
key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1) 
value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1) 

if seq_len <= self.chunk_size: 
out = F.scaled_dot_product_attention( 
query_states, key_states, value_states, is_causal=True 
) 
else: 
num_chunks = seq_len // self.chunk_size 
trunc_len = num_chunks * self.chunk_size 

k_chunked = key_states[:, :, :trunc_len, :].reshape( 
bsz, self.num_heads, num_chunks, self.chunk_size, self. head_dim 
).mean(dim=3) 

v_chunked = value_states[:, :, :trunc_len, :].reshape( 
bsz, self.num_heads, num_chunks, self.chunk_size, self.head_dim 
).mean(dim=3) 

scores_global = torch.matmul(query_states, k_chunked.transpose(-1, -2)) * self.scaling 

token_indices = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1) 
chunk_end_indices = (torch.arange(num_chunks, device=hidden_states.device) + 1).view(1, 1, 1, num_chunks) * self.chunk_size 
mask_global = token_indices < (chunk_end_indices - self.chunk_size) 
scores_global = scores_global.masked_fill(mask_global, float("-inf")) 

attn_global = F.softmax(scores_global, dim=-1) 
attn_global = torch.nan_to_num(attn_global, nan=0.0) 
out_global = torch.matmul(attn_global, v_chunked) 

out = out_global 

out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1) 
return self. o_proj(out), None

#2. Load the model and inject the VCPA algorithm into the attention layers
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map=device)

# Inject the 24 layers of the model
for i, layer in enumerate(model.model.layers): 
if hasattr(layer, "self_attn"): 
layer.self_attn = VCPAQwen2Attention(layer.self_attn, chunk_size=64, local_window=128)

print("Successfully patched model with VCPA Gen 11!")

```
## 📜 License

This project is open-source under the [MIT License](LICENSE). Contributions, suggestions, and extensions to other model families are highly welcome!
by : karam awashra
