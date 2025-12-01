import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# --- 1. DEFINE CLASSES ---

class MultiBaseQwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, num_heads=12):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        min_base, max_base = 100.0, 500000.0
        
        self.bases = torch.logspace(math.log10(min_base), math.log10(max_base), num_heads, device=device).unsqueeze(1)
        inv_freq = 1.0 / (self.bases ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("hd,s->hsd", self.inv_freq, t)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype), self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype))

class DummyRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, num_heads=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, seq_len=None):
        # Always return Identity (Pos 0)
        shape = (1, 1, seq_len, self.dim)
        return (torch.ones(shape, device=x.device, dtype=x.dtype), torch.zeros(shape, device=x.device, dtype=x.dtype))

# --- 2. SURGERY FUNCTION ---

def inject_rope_surgery(model, mode="psystem"):
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    print(f"\n💉 INJECTING: {mode.upper()} ROPE...")
    
    for i, layer in enumerate(model.model.layers):
        if mode == "psystem":
            new_rope = MultiBaseQwen2RotaryEmbedding(head_dim, config.max_position_embeddings, config.rope_theta, model.device, num_heads)
        elif mode == "dummy":
            new_rope = DummyRotaryEmbedding(head_dim, config.max_position_embeddings, device=model.device)
        
        layer.self_attn.rotary_emb = new_rope
        
    print("✅ Surgery Complete.\n")
    return model

# --- 3. TRAINING LOOP ---

def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    
    # LOAD MODEL
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")

    # !!! CHOOSE MODE HERE !!! 
    # Change to 'psystem' for your real experiment
    # Change to 'dummy' to verify patching works (Loss should be bad)
    SURGERY_MODE = "psystem" 
    
    model = inject_rope_surgery(model, mode=SURGERY_MODE)

    # LORA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    model = get_peft_model(model, peft_config)

    # DATASET
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:500]") # Small subset for quick check
    
    def format_prompt(sample):
        conversations = sample['conversations']
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        text = ""
        for turn in conversations:
            text += f"<|im_start|>{role_map.get(turn['from'], turn['from'])}\n{turn['value']}<|im_end|>\n"
        return text + "<|im_start|>assistant\n"

    # TRAINER
    training_args = SFTConfig(
        output_dir=f"./qwen-{SURGERY_MODE}-check",
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=1,
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

    print(f"🚀 Launching {SURGERY_MODE.upper()} sanity check...")
    trainer.train()

if __name__ == "__main__":
    train()