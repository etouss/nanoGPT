import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. THE AGGRESSIVE DUMMY
class NuclearDummyRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        print(f"👻 NuclearDummy Initialized (Dim: {dim})")

    def forward(self, x, seq_len=None, position_ids=None, **kwargs):
        # IF THIS PRINTS, WE WIN.
        print(f"\n\n >>> ☢️ NUCLEAR SUCCESS: Dummy RoPE called with shape {x.shape}! ☢️ <<< \n\n")
        raise RuntimeError("✅ SURGERY SUCCESSFUL: The model tried to use my custom RoPE!")
        
        # Fallback for shape compatibility if we didn't raise
        shape = (1, 1, 2048, self.dim) # Huge buffer to be safe
        return (torch.ones(shape, device=x.device, dtype=x.dtype), 
                torch.zeros(shape, device=x.device, dtype=x.dtype))

# 2. RECURSIVE REPLACER
def replace_rope_recursively(module, target_cls_name="RotaryEmbedding"):
    """
    Traverses the module tree. If it finds a child that looks like RoPE, it kills it.
    """
    count = 0
    for name, child in module.named_children():
        # Check if this child is a Rotary Embedding
        if "RotaryEmbedding" in child.__class__.__name__:
            print(f"   ⚔️ Found target: {name} ({child.__class__.__name__})")
            
            # Create Replacement
            # Try to grab dim from the existing child
            dim = getattr(child, "dim", 64) 
            new_rope = NuclearDummyRotaryEmbedding(dim, device=child.inv_freq.device if hasattr(child, 'inv_freq') else None)
            
            # SWAP
            setattr(module, name, new_rope)
            count += 1
        else:
            # Recurse
            count += replace_rope_recursively(child, target_cls_name)
    return count

def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("📥 Loading Model (CPU, Eager)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=None, 
        torch_dtype=torch.float32, # Use float32 on CPU to avoid potential bfloat16 errors
        attn_implementation="eager"
    )

    print("\n🔍 STARTING NUCLEAR SEARCH & DESTROY...")
    # We search the ENTIRE model structure, not just layers
    replacements = replace_rope_recursively(model)
    print(f"✅ Replaced {replacements} RotaryEmbedding instances globally.")
    
    if replacements == 0:
        print("❌ ERROR: No RotaryEmbeddings found! The class name might be different.")
        # Let's inspect the model structure to find the real name
        print(model)
        return

    print("🚚 Moving to CUDA...")
    model.to("cuda")

    # Add a hook to the first attention layer to see if IT runs
    # Find first layer
    base_model = getattr(model, "model", model)
    if hasattr(base_model, "layers"):
        first_layer = base_model.layers[0].self_attn
        def hook_fn(module, input, output):
            print("👁️ HOOK: Attention Layer 0 Forward Pass Started!")
        first_layer.register_forward_hook(hook_fn)
        print("🪝 Forward hook attached to Attention Layer 0")

    # Setup dummy training
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:10]") 
    def format_prompt(sample):
        conversations = sample['conversations']
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        text = ""
        for turn in conversations:
            text += f"<|im_start|>{role_map.get(turn['from'], turn['from'])}\n{turn['value']}<|im_end|>\n"
        return text + "<|im_start|>assistant\n"

    # Minimal Trainer
    training_args = SFTConfig(
        output_dir="./nuclear-debug",
        dataset_text_field="text",
        max_length=64,
        per_device_train_batch_size=1,
        max_steps=1,
        logging_steps=1,
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
    try:
        trainer.train()
    except RuntimeError as e:
        if "SURGERY SUCCESSFUL" in str(e):
            print(f"\n\n🏆 MISSION ACCOMPLISHED: {e}")
        else:
            print(f"\n\n💥 CRASH (Unrelated): {e}")
    except Exception as e:
        print(f"\n\n❓ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    train()