import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# 1. THE EUKARYOTIC ROPE (Heterogeneous Catalyst)
# ==============================================================================
class MultiBaseQwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, num_heads=14):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.num_heads = num_heads

        # P-SYSTEM LOGIC: Distribute Bases
        # We go from 100.0 (High Viscosity/Local) to 500,000.0 (Superfluid/Global)
        min_base = 100.0
        max_base = 500000.0
        
        # Calculate distinct base for each head
        # Shape: [num_heads, 1]
        self.bases = torch.logspace(
            math.log10(min_base), 
            math.log10(max_base), 
            num_heads, 
            device=device
        ).unsqueeze(1)
        
        # Compute frequencies
        # dim_indices: [0, 2, 4... dim]
        dim_indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        self.inv_freq = 1.0 / (self.bases ** (dim_indices / dim))
        
        # Register buffer (not a parameter)
        self.register_buffer("inv_freq_buffer", self.inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq_buffer.dtype)

        # Outer Product: [num_heads, dim/2] x [seq_len] -> [num_heads, seq_len, dim/2]
        # Reshape to [num_heads, seq_len, dim/2]
        freqs = torch.einsum("hd,s->hsd", self.inv_freq_buffer, t)
        
        # Create full embedding (cos/sin repeated)
        emb = torch.cat((freqs, freqs), dim=-1) # [H, S, D]
        
        # Reshape to [1, H, S, D] for broadcasting against [B, H, S, D]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).to(dtype), persistent=False)

    def forward(self, x, seq_len=None, position_ids=None, **kwargs):
        # Robustly determine sequence length
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.shape[-1]
            else:
                # Fallback if x is [Batch, Seq, Hidden] (as seen in debug)
                # or [Batch, Head, Seq, Dim]
                seq_len = x.shape[-2] if x.dim() == 4 else x.shape[1]

        if self.cos_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Return cached slices
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# ==============================================================================
# 2. RECURSIVE SURGERY (The Proven Method)
# ==============================================================================
def replace_rope_recursively(module, num_heads, head_dim):
    count = 0
    for name, child in module.named_children():
        if "RotaryEmbedding" in child.__class__.__name__:
            print(f"   ⚔️ Replacing: {name} ({child.__class__.__name__})")
            
            # Inject MultiBase RoPE
            new_rope = MultiBaseQwen2RotaryEmbedding(
                dim=head_dim,
                # Try to inherit existing config if possible, else defaults
                max_position_embeddings=getattr(child, "max_position_embeddings", 2048),
                device=getattr(child, "inv_freq", torch.tensor(0)).device, 
                num_heads=num_heads
            )
            setattr(module, name, new_rope)
            count += 1
        else:
            count += replace_rope_recursively(child, num_heads, head_dim)
    return count

# ==============================================================================
# 3. MAIN TRAINING LOOP
# ==============================================================================
def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # A. Robust Loading (CPU + Eager)
    print("📥 Loading Model on CPU (Eager Mode)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=None,             # Disable Accelerate auto-map
        torch_dtype=torch.float32,   # Use float32 on CPU for safety
        attn_implementation="eager"  # Force Python implementation
    )

    # B. Surgery
    print("🔬 Starting P-System Surgery...")
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    replaced_count = replace_rope_recursively(model, num_heads, head_dim)
    print(f"✅ Replaced {replaced_count} RotaryEmbedding instances.")
    
    # Qwen-0.5B has 24 layers. We expect 24 replacements.
    if replaced_count != config.num_hidden_layers:
        print(f"⚠️ WARNING: Expected {config.num_hidden_layers} replacements, but got {replaced_count}.")
    
    # C. Move to GPU
    print("🚚 Moving model to GPU...")
    model.to("cuda")

    # D. Apply LoRA
    print("🧬 Applying LoRA Adapters...")
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

    # E. Data
    print("📚 Loading OpenHermes...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:5000]") 
    
    def format_prompt(sample):
        conversations = sample['conversations']
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        text = ""
        for turn in conversations:
            text += f"<|im_start|>{role_map.get(turn['from'], turn['from'])}\n{turn['value']}<|im_end|>\n"
        return text + "<|im_start|>assistant\n"

    # F. Trainer
    # SFTConfig setup (trl >= 0.25.0 style)
    training_args = SFTConfig(
        output_dir="./qwen-psystem-rope-real",
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, # Higher LR for structural adaptation
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=200,
        fp16=False,
        bf16=True, # Switch back to BF16 for GPU training
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
    
    print("💾 Saving Final Model...")
    trainer.save_model("./qwen-psystem-rope-real-final")

if __name__ == "__main__":
    train()