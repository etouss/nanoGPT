import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# ==============================================================================
# 1. THE HETEROGENEOUS CATALYST (Multi-Base RoPE)
# ==============================================================================

class MultiBaseQwen2RotaryEmbedding(nn.Module):
    """
    P-System 'Eukaryotic' RoPE.
    Instead of a monolithic base (e.g. 10000) for all heads, we distribute 
    bases geometrically across heads to create specialized chemical environments.
    
    Head 0: High Viscosity (Base ~100) -> Strictly Local / Syntax
    Head N: Superfluid (Base ~1M) -> Global / Semantic
    """
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
        # Shape: [num_heads, 1]
        self.bases = torch.logspace(
            math.log10(min_base), 
            math.log10(max_base), 
            num_heads, 
            device=device
        ).unsqueeze(1)
        
        # Standard RoPE frequency calculation, but broadcasted per head
        # dim_indices: [0, 2, 4... dim]
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
        # We need to reshape carefully for broadcasting
        # inv_freq: [H, D/2]
        # t: [S]
        # freqs: [H, S, D/2]
        freqs = torch.einsum("hd,s->hsd", self.inv_freq, t)
        
        # Create full embedding (cos/sin repeated)
        emb = torch.cat((freqs, freqs), dim=-1) # [H, S, D]
        
        # Reshape to [1, H, S, D] for broadcasting against [B, H, S, D]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Return the cached cos/sin sliced to the current sequence length
        # Shape: [1, num_heads, seq_len, dim]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# ==============================================================================
# 2. THE SURGERY (Architecture Injection)
# ==============================================================================

def inject_psystem_rope(model):
    """
    Replaces the standard Qwen2RotaryEmbedding with our MultiBase version.
    """
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    print(f"🔬 P-System Surgery: Injecting Heterogeneous Catalysts into {len(model.model.layers)} layers...")
    print(f"   - Head Dim: {head_dim}")
    print(f"   - Num Heads: {num_heads}")
    
    for i, layer in enumerate(model.model.layers):
        # Initialize our new "Eukaryotic" RoPE
        new_rope = MultiBaseQwen2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta, # Ignored, overwritten by our logic
            device=model.device,
            num_heads=num_heads
        )
        
        # Swap it out
        layer.self_attn.rotary_emb = new_rope
        
    print("✅ Surgery Complete. Attention heads are now specialized.")
    return model

# ==============================================================================
# 3. TRAINING SETUP
# ==============================================================================

def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # A. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token # Qwen doesn't have a pad token by default
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" # Required for custom RoPE injection compatibility
    )

    # B. Perform Surgery (Before LoRA!)
    #model = inject_psystem_rope(model)

    # C. Apply LoRA
    # We target q_proj and k_proj heavily because they interact with the Position Catalyst.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,                # High rank to allow significant re-alignment of physics
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # D. Load Dataset
    # OpenHermes 2.5 is great for reasoning. We take a small subset for a fast demo.
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:5000]") 
    
    def format_prompt(sample):
        # Qwen chat format
        return f"<|im_start|>user\n{sample['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{sample['output']}\n<|im_end|>"

    # E. Trainer
    training_args = TrainingArguments(
        output_dir="./qwen-psystem-rope-adapter",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, # Slightly higher LR for structural adaptation
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=1000,
        fp16=False,
        bf16=True, # Recommended for Qwen
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=format_prompt,
        max_seq_length=1024,
        args=training_args,
    )

    print("🚀 Launching P-System Training...")
    trainer.train()
    
    print("💾 Saving Adapter...")
    trainer.save_model("./qwen-psystem-rope-final")

if __name__ == "__main__":
    train()