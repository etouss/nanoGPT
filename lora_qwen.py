import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, repeat_kv
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# 0. HELPER FUNCTIONS (Manual Math)
# ==============================================================================
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def manual_apply_rope(x, cos, sin):
    """
    Explicit RoPE application to avoid transformers helper magic.
    Expects x: [B, H, S, D]
    Expects cos, sin: [1, H, S, D]
    """
    return (x * cos) + (rotate_half(x) * sin)

# ==============================================================================
# 1. THE "HEMISPHERE" ROPE (GQA Optimized)
# ==============================================================================
class HemisphereQwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, config=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        self.num_kv_heads = config.num_key_value_heads # 2
        self.num_q_heads = config.num_attention_heads  # 14
        self.group_size = self.num_q_heads // self.num_kv_heads # 7

        # P-SYSTEM LOGIC: 2 Physics Environments
        min_base = 100.0
        max_base = 1000000.0
        
        # Group 0 -> 100.0, Group 1 -> 1,000,000.0
        bases_kv = torch.tensor([min_base, max_base], device=device).view(2, 1) # [2, 1]
        
        self.bases_q = bases_kv.repeat_interleave(self.group_size, dim=0) # [14, 1]
        self.bases_k = bases_kv # [2, 1]

        # Compute Frequencies
        dim_indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        term = dim_indices / dim
        
        _inv_freq_q = 1.0 / (self.bases_q ** term) # [14, Dim/2]
        _inv_freq_k = 1.0 / (self.bases_k ** term) # [2, Dim/2]
        
        self.register_buffer("inv_freq_q", _inv_freq_q, persistent=False)
        self.register_buffer("inv_freq_k", _inv_freq_k, persistent=False)
        
        self.register_buffer("cos_q", None, persistent=False)
        self.register_buffer("sin_q", None, persistent=False)
        self.register_buffer("cos_k", None, persistent=False)
        self.register_buffer("sin_k", None, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq_q.dtype)

        # 1. Compute for Queries [14 Heads]
        freqs_q = torch.einsum("hd,s->hsd", self.inv_freq_q, t)
        emb_q = torch.cat((freqs_q, freqs_q), dim=-1) # [14, S, D]
        
        # 2. Compute for Keys [2 Heads]
        freqs_k = torch.einsum("hd,s->hsd", self.inv_freq_k, t)
        emb_k = torch.cat((freqs_k, freqs_k), dim=-1) # [2, S, D]

        # Cache with broadcasting shapes: [1, H, S, D]
        self.register_buffer("cos_q", emb_q.cos().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_q", emb_q.sin().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("cos_k", emb_k.cos().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_k", emb_k.sin().unsqueeze(0).to(dtype), persistent=False)

    def forward(self, x, seq_len=None, position_ids=None, **kwargs):
        # Robust seq_len extraction
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.shape[-1]
            else:
                seq_len = x.shape[1]
        
        if torch.is_tensor(seq_len): seq_len = seq_len.item()

        if self.cos_q is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_q[:, :, :seq_len, ...], self.sin_q[:, :, :seq_len, ...],
            self.cos_k[:, :, :seq_len, ...], self.sin_k[:, :, :seq_len, ...]
        )

# ==============================================================================
# 2. MONKEY PATCH FORWARD (Manual Rotation)
# ==============================================================================
def psystem_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    bsz, q_len, _ = hidden_states.size()

    # FIX: Robustly get attributes that might be missing on 'self'
    num_heads = getattr(self, 'num_heads', self.config.num_attention_heads)
    num_key_value_heads = getattr(self, 'num_key_value_heads', self.config.num_key_value_heads)
    
    # Calculate head_dim if missing
    if hasattr(self, 'head_dim'):
        head_dim = self.head_dim
    else:
        head_dim = self.config.hidden_size // num_heads

    # FIX: Robustly get hidden_size (The source of your crash)
    if hasattr(self, 'hidden_size'):
        hidden_size = self.hidden_size
    else:
        hidden_size = self.config.hidden_size

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # --- P-SYSTEM LOGIC ---
    cos_q, sin_q, cos_k, sin_k = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # MANUAL APPLICATION (Safe from transformers helpers)
    query_states = manual_apply_rope(query_states, cos_q, sin_q)
    key_states = manual_apply_rope(key_states, cos_k, sin_k)
    # ----------------------

    if past_key_value is not None:
        cache_kwargs = {"sin": sin_k, "cos": cos_k}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # FIX: Use the robust hidden_size variable we defined at the top
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

# ==============================================================================
# 3. SURGERY FUNCTION
# ==============================================================================
def inject_psystem_surgery(model):
    print("🔬 P-System Surgery: Patching Qwen2Attention...")
    
    Qwen2Attention.forward = psystem_forward
    print("   ✅ Monkey-patched Qwen2Attention.forward")

    config = model.config
    
    # Unwrap model
    base_model = getattr(model, "model", model)
    if hasattr(base_model, "model"): base_model = base_model.model

    count = 0
    for layer in base_model.layers:
        new_rope = HemisphereQwen2RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            device=model.device,
            config=config
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
    
    print("📥 Loading Model (CPU/Eager)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=None,
        torch_dtype=torch.float32, 
        attn_implementation="eager"
    )

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

    training_args = SFTConfig(
        output_dir="./qwen-psystem-hemisphere",
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=200,
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

    print("🚀 Launching Hemisphere Training...")
    trainer.train()
    trainer.save_model("./qwen-psystem-hemisphere-final")

if __name__ == "__main__":
    train()