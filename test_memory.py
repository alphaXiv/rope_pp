"""Test memory requirements before full training"""
import torch
from rope_pp.configuration_llama import LlamaConfig
from rope_pp.modeling_llama_rope_pp import LlamaForCausalLM

# Test configuration
rope_config = {'imag': True, 'imag_mode': 'imag2'}

config = LlamaConfig.from_pretrained('configs/rope-376m-config.json')
config.gradient_checkpointing = True
config.use_cache = False
config._attn_implementation = "flash_attention_2"
config.torch_dtype = torch.bfloat16
config.rope_config = rope_config

print("Creating model...")
model = LlamaForCausalLM(config=config)
model = model.to('cuda')
model.train()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Test forward pass with different batch sizes
batch_sizes = [1, 2, 4, 8, 16]
seq_length = 4096

for bs in batch_sizes:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy batch
        input_ids = torch.randint(0, config.vocab_size, (bs, seq_length), device='cuda')
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Check memory
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✓ Batch size {bs:2d}, seq_len {seq_length}: {mem_allocated:.2f} GB")
        
        # Cleanup
        del input_ids, labels, outputs, loss
        model.zero_grad()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ Batch size {bs:2d}, seq_len {seq_length}: OOM")
            torch.cuda.empty_cache()
        else:
            raise e

print("\nRecommendation:")
max_working_bs = 1
for bs in batch_sizes:
    try:
        torch.cuda.empty_cache()
        input_ids = torch.randint(0, config.vocab_size, (bs, seq_length), device='cuda')
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()
        max_working_bs = bs
        del input_ids, labels, outputs
        model.zero_grad()
    except:
        break

print(f"Maximum safe batch size: {max_working_bs}")
print(f"Recommended settings:")
print(f"  batch_size = {max_working_bs}")
print(f"  gradient_accumulation_steps = {128 // max_working_bs}")
print(f"  Effective batch size = {max_working_bs * (128 // max_working_bs)}")
