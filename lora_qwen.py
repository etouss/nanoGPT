import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. ROBUST DUMMY CLASS (The Scream Test)
class DummyRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, seq_len=None):
        # IF THIS PRINTS, WE WIN.
        print("\n\n >>> 🚨 SURGERY SUCCESSFUL: DUMMY ROPE CALLED! 🚨 <<< \n\n")
        raise RuntimeError("✅ SURGERY SUCCESSFUL: The model tried to use my custom RoPE!")
        
        # Unreachable code, but keeps shape correct just in case
        shape = (1, 1, seq_len, self.dim)
        return (torch.ones(shape, device=x.device, dtype=x.dtype), 
                torch.zeros(shape, device=x.device, dtype=x.dtype))

# 2. SURGERY FUNCTION
def inject_rope_surgery(model):
    print(f"\n🔬 STARTING SURGERY...")
    
    # Unwrap PEFT if present to find the base model
    base_model = getattr(model, "model", model)
    if hasattr(base_model, "model"):
        base_model = base_model.model
        
    # Check what Attention implementation is being used
    first_layer = base_model.layers[0]
    attn_type = type(first_layer.self_attn).__name__
    print(f"🧐 Detected Attention Implementation: {attn_type}")
    
    if "Flash" in attn_type or "Sdpa" in attn_type:
        print("⚠️ WARNING: Flash/SDPA Attention detected. These optimized kernels might bypass custom modules!")
        print("   We will force Eager Execution in the loader.")

    for i, layer in enumerate(base_model.layers):
        # Create Dummy
        head_dim = layer.self_attn.head_dim
        new_rope = DummyRotaryEmbedding(head_dim, device=base_model.device)
        
        # KEY: Force the replacement
        layer.self_attn.rotary_emb = new_rope
    
    print("✅ Surgery Complete. All Rotary Embeddings are now DUMMY.\n")
    return model

def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    
    # --- FIX 1: FORCE EAGER & CPU LOAD ---
    print("📥 Loading Model on CPU (Eager Mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=None,             # <--- DISABLE ACCELERATE AUTO DEVICE MAP
        torch_dtype=torch.bfloat16, 
        attn_implementation="eager"  # <--- FORCE PYTHON PATH
    )

    # --- FIX 2: APPLY SURGERY BEFORE PEFT & DEVICE MOVE ---
    #model = inject_rope_surgery(model)

    # --- FIX 3: MOVE TO GPU MANUALLY ---
    print("🚚 Moving model to GPU...")
    model.to("cuda")

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Data
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:50]") 
    
    def format_prompt(sample):
        conversations = sample['conversations']
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        text = ""
        for turn in conversations:
            text += f"<|im_start|>{role_map.get(turn['from'], turn['from'])}\n{turn['value']}<|im_end|>\n"
        return text + "<|im_start|>assistant\n"

    # Trainer
    training_args = SFTConfig(
        output_dir="./qwen-dummy-debug",
        dataset_text_field="text",
        max_length=128,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        max_steps=5,     # Only run a few steps to trigger the error
        logging_steps=1,
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

    print("🚀 Launching Training (Expect Crash)...")
    try:
        trainer.train()
    except RuntimeError as e:
        print(f"\n\n🏆 SUCCESS! The model crashed with your error: {e}")
    except Exception as e:
        print(f"\n\n❌ FAILED with unexpected error: {e}")

if __name__ == "__main__":
    train()