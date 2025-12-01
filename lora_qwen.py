import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# 1. THE HETEROGENEOUS CATALYST (Multi-Base RoPE)
# ==============================================================================

class MultiBaseQwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, num_heads=12):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.num_heads = num_heads

        # P-SYSTEM LOGIC: Distribute Bases
        # We go from 100.0 (Local) to 500,000.0 (Global)
        min_base = 100.0
        max_base = 500000.0
        
        # Calculate distinct base for each head
        self.bases = torch.logspace(
            math.log10(min_base), 
            math.log10(max_base), 
            num_heads, 
            device=device
        ).unsqueeze(1)
        
        # Standard RoPE frequency calculation, but broadcasted per head
        inv_freq = 1.0 / (self.bases ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=self.inv_freq.device, 
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # Outer Product: [num_heads, dim/2] x [seq_len] -> [num_heads, seq_len, dim/2]
        # Reshape to [num_heads, seq_len, dim/2]
        freqs = torch.einsum("hd,s->hsd", self.inv_freq, t)
        
        # Create full embedding (cos/sin repeated)
        emb = torch.cat((freqs, freqs), dim=-1) # [H, S, D]
        
        # Reshape to [1, H, S, D] for broadcasting against [B, H, S, D]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# ==============================================================================
# 2. THE SURGERY (Architecture Injection)
# ==============================================================================

def inject_psystem_rope(model):
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    print(f"🔬 P-System Surgery: Injecting Heterogeneous Catalysts into {len(model.model.layers)} layers...")
    
    for i, layer in enumerate(model.model.layers):
        new_rope = MultiBaseQwen2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            device=model.device,
            num_heads=num_heads
        )
        layer.self_attn.rotary_emb = new_rope
        
    print("✅ Surgery Complete.")
    return model

# ==============================================================================
# 3. TRAINING SETUP
# ==============================================================================

def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # A. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )

    # B. Perform Surgery
    model = inject_psystem_rope(model)

    # C. Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,                
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # D. Load Dataset
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:5000]") 
    
    # --- FIX IS HERE: Correct format_prompt for ShareGPT structure ---
    def format_prompt(sample):
        # OpenHermes-2.5 uses 'conversations' with 'from' and 'value' keys
        conversations = sample['conversations']
        
        # Map ShareGPT roles to Qwen/ChatML roles
        role_map = {
            "human": "user",
            "gpt": "assistant",
            "system": "system"
        }
        
        text = ""
        for turn in conversations:
            role = role_map.get(turn['from'], turn['from'])
            content = turn['value']
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
        return text + "<|im_start|>assistant\n"

    # E. Trainer Setup
    # Updated for trl >= 0.25.0 compatibility
    training_args = SFTConfig(
        output_dir="./qwen-psystem-rope-adapter",
        dataset_text_field="text",
        max_length=1024,              # 'max_length' replaces 'max_seq_length' in Config
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

    print("🚀 Launching P-System Training...")
    trainer.train()
    
    print("💾 Saving Adapter...")
    trainer.save_model("./qwen-psystem-rope-final")

if __name__ == "__main__":
    train()