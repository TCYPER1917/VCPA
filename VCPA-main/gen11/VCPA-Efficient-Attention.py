import os
import gc
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 4096  # تم رفع السياق إلى 4096 لاختبار كسر حدود VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

# --- Generation 11 Novel Algorithm: Vectorized Chunk-Pooled Attention (VCPA) ---
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
        
        self.chunk_size = chunk_size
        self.local_window = local_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple = None,
        attention_mask: torch.Tensor = None,
        past_key_values = None,
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
                bsz, self.num_heads, num_chunks, self.chunk_size, self.head_dim
            ).mean(dim=3)
            
            v_chunked = value_states[:, :, :trunc_len, :].reshape(
                bsz, self.num_heads, num_chunks, self.chunk_size, self.head_dim
            ).mean(dim=3)

            scores_global = torch.matmul(query_states, k_chunked.transpose(-1, -2)) * self.scaling
            
            token_indices = torch.arange(seq_len, device=DEVICE).view(1, 1, seq_len, 1)
            chunk_end_indices = (torch.arange(num_chunks, device=DEVICE) + 1).view(1, 1, 1, num_chunks) * self.chunk_size
            mask_global = token_indices < (chunk_end_indices - self.chunk_size)
            scores_global = scores_global.masked_fill(mask_global, float("-inf"))
            
            attn_global = F.softmax(scores_global, dim=-1)
            attn_global = torch.nan_to_num(attn_global, nan=0.0)
            out_global = torch.matmul(attn_global, v_chunked)

            out = out_global

        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        return self.o_proj(out), None

def run_benchmark():
    results = {}
    print(f"Loading {MODEL_ID} on {DEVICE} for SEQ_LEN={SEQ_LEN}...")
    
    # 1. Baseline Model (مع معالجة خطأ الـ OOM في حال حدث عند 4096)
    reset_memory()
    try:
        model_baseline = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=DEVICE
        )
        model_baseline.train()
        input_ids = torch.randint(0, 1000, (1, SEQ_LEN), device=DEVICE)
        
        # القياس
        start_time = time.time()
        out = model_baseline(input_ids)
        loss = out.logits.sum()
        loss.backward()
        baseline_time = time.time() - start_time
        baseline_vram = get_peak_memory()
        
        results["baseline"] = {
            "time_s": baseline_time,
            "peak_vram_mb": baseline_vram
        }
        print(f"Baseline (4096): Time={baseline_time:.4f}s, Peak VRAM={baseline_vram:.2f} MB")
        del model_baseline
    except torch.cuda.OutOfMemoryError:
        results["baseline"] = {"error": "OutOfMemoryError (OOM) at 4096"}
        print("Baseline crashed with OOM at 4096 tokens!")
    
    reset_memory()
    
    # 2. Optimized Model (Gen 11 Surgery at 4096)
    model_opt = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    for i, layer in enumerate(model_opt.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn = VCPAQwen2Attention(layer.self_attn, chunk_size=64, local_window=128)
            
    model_opt.train()
    reset_memory()
    
    input_ids = torch.randint(0, 1000, (1, SEQ_LEN), device=DEVICE)
    start_time = time.time()
    out = model_opt(input_ids)
    loss = out.logits.sum()
    loss.backward()
    opt_time = time.time() - start_time
    opt_vram = get_peak_memory()
    
    results["optimized_gen11_4096"] = {
        "time_s": opt_time,
        "peak_vram_mb": opt_vram
    }
    print(f"Gen 11 (4096): Time={opt_time:.4f}s, Peak VRAM={opt_vram:.2f} MB")
    
    with open("results_4096.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Saved results_4096.json")

if __name__ == "__main__":
    run_benchmark()