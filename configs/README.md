# Configuration Files

This directory contains configuration files for different GPT versions and training scenarios.

## Version-Specific Configurations

- **gpt2_config.py**: Configuration for GPT-2 models (124M-1.5B parameters)
- **gpt3_config.py**: Configuration for GPT-3 models (175B parameters)
- **gpt3.5_config.py**: Configuration for GPT-3.5 models (~350B parameters)
- **gpt4_config.py**: Configuration for GPT-4 models with MoE architecture
- **gpt4.5_config.py**: Configuration for GPT-4.5 models with enhanced MoE
- **gpt5_config.py**: Configuration for theoretical GPT-5 models with unlimited context

## Configuration Parameters

### Common Parameters

- **block_size**: Maximum context length
- **vocab_size**: Vocabulary size
- **n_layer**: Number of transformer layers
- **n_head**: Number of attention heads
- **n_embd**: Embedding dimension
- **dropout**: Dropout probability
- **bias**: Whether to use bias in Linear and LayerNorm layers

### MoE-Specific Parameters

- **use_moe**: Whether to use Mixture of Experts
- **num_experts**: Total number of experts
- **num_active_experts**: Number of experts active per token
- **expert_capacity_factor**: Capacity factor for load balancing

### Unlimited Context Parameters

- **billion_token_context**: Enable billion-token context support
- **use_sparse_attention**: Use sparse attention mechanisms
- **preserve_begin_tokens**: Number of beginning tokens to preserve
- **preserve_end_tokens**: Number of end tokens to preserve
- **token_eviction_strategy**: Strategy for evicting tokens

## Usage

Configurations are loaded through the version manager:

```python
from utils.version_manager import load_model_config

# Load configuration for a specific version
config = load_model_config('gpt4')

# Access and modify configuration parameters
print(f"Context length: {config.block_size}")
config.block_size = 16384  # Override context length
```

For training, configurations can be specified as command-line arguments:

```bash
python train.py --config configs/gpt4_config.py --block_size=32768
```
