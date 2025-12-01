import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import types

# ==============================================================================
# 1. THE EUKARYOTIC ROPE (GQA Compatible)
# ==============================================================================
class MultiBaseQwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, num_heads=14, num_kv_heads=2):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.num_heads = num_heads       # 14
        self.num_kv_heads = num_kv_heads # 2

        # P-SYSTEM LOGIC: Distribute Bases
        # We generate 14 unique bases for the 14 Query Heads
        min_base = 100.0
        max_base = 500000.0
        
        # Shape: [14, 1]
        self.bases = torch.logspace(
            math.log10(min_base), 
            math.log10(max_base), 
            num_heads, 
            device=device
        ).unsqueeze(1)
        
        # Compute frequencies for Q [14 heads]
        dim_indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        self.inv_freq = 1.0 / (self.bases ** (dim_indices / dim))
        
        self.register_buffer("inv_freq_buffer", self.inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq_buffer.dtype)

        # Outer Product: [14, D/2] x [S] -> [14, S, D/2]
        freqs = torch.einsum("hd,s->hsd", self.inv_freq_buffer, t)
        emb = torch.cat((freqs, freqs), dim=-1) # [14, S, D]
        
        # Cache Q-compatible shape: [1, 14, S, D]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).to(dtype), persistent=False)

    def forward(self, x, seq_len=None, position_ids=None, **kwargs):
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.shape[-1]
            else:
                seq_len = x.shape[1]
        
        if torch.is_tensor(seq_len): seq_len = seq_len.item()

        if self.cos_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Return the full 14-head tensor
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

# ==============================================================================
# 2. CUSTOM FORWARD METHOD (The Monkey Patch)
# ==============================================================================
def psystem_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    # This is a Modified Qwen2Attention.forward to handle Heterogeneous RoPE + GQA
    
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # --- P-SYSTEM LOGIC STARTS HERE ---
    # 1. Get the 14-head Rotation Tensors [1, 14, S, D]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    # 2. Apply to Query (Direct match: 14 vs 14) -> OK
    # query_states: [B, 14, S, D] * cos: [1, 14, S, D]
    query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin) # Ignore K output here

    # 3. Apply to Key (Mismatch: 2 vs 14) -> MANUAL FIX
    # We must downsample the 14-head physics to the 2-head physics.
    # Logic: KV-Head 0 serves Q-Heads 0-6. Let's assign KV-Head 0 the physics of Q-Head 0 (or average).
    # Simplest P-System Logic: The KV head adopts the Base of the *First* Q-head in its group.
    
    # ratio = 14 // 2 = 7
    ratio = self.num_heads // self.num_key_value_heads
    
    # Slice cos/sin to get only the representative heads (Indices: 0, 7)
    # shape: [1, 2, S, D]
    cos_k = cos[:, ::ratio, :, :]
    sin_k = sin[:, ::ratio, :, :]
    
    # key_states: [B, 2, S, D] * cos_k: [1, 2, S, D] -> OK
    # We use apply_rotary_pos_emb purely for the math helper, passing dummy Q
    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos_k, sin_k) 
    # --- P-SYSTEM LOGIC ENDS ---

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos} # Pass original for cache consistency if needed
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Repeat KV for GQA
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Standard Attention ...
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

# ==============================================================================
# 3. SURGERY FUNCTION
# ==============================================================================
def inject_psystem_surgery(model):
    print("🔬 P-System Surgery: Patching Qwen2Attention...")
    
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    # 1. Replace the Method on the Class (Affects all instances)
    # This is safer than replacing instances one by one
    Qwen2Attention.forward = psystem_forward
    print("   ✅ Monkey-patched Qwen2Attention.forward")

    # 2. Replace the Rotary Embedding Modules
    count = 0
    for layer in model.model.layers:
        new_rope = MultiBaseQwen2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            device=model.device,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        layer.self_attn.rotary_emb = new_rope
        count += 1
        
    print(f"   ✅ Replaced {count} Rotary Modules.")
    return model

# ==============================================================================
# 4. MAIN TRAIN
# ==============================================================================
def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("📥 Loading Model (CPU / Eager)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=None,
        torch_dtype=torch.float32, 
        attn_implementation="eager" # Required for our custom forward to run
    )

    # Apply Surgery
    model = inject_psystem_surgery(model)
    
    print("🚚 Moving to CUDA...")
    model.to("cuda")

    # LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Data
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:2000]") 
    
    def format_prompt(sample):
        conversations = sample['conversations']
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        text = ""
        for turn in conversations:
            text += f"<|im_start|>{role_map.get(turn['from'], turn['from'])}\n{turn['value']}<|im_end|>\n"
        return text + "<|im_start|>assistant\n"

    # Trainer
    training_args = SFTConfig(
        output_dir="./qwen-psystem-gqa",
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=100,
        fp16=False,
        bf16=True,
        report_to="none",
        packing=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=format_prompt,
        args=training_args
    )

    print("🚀 Launching Training...")
    trainer.train()
    trainer.save_model("./qwen-psystem-gqa-final")

if __name__ == "__main__":
    train()