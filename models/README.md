# Models

This directory contains model implementations for different GPT versions.

## Overview

Each GPT version has its own model implementation with version-specific optimizations:

- **gpt2_model.py**: GPT-2 implementation with standard transformer architecture
- **gpt3_model.py**: GPT-3 with scaled transformer architecture
- **gpt3.5_model.py**: GPT-3.5 with optimized transformer and instruction tuning
- **gpt4_model.py**: GPT-4 with mixture of experts architecture
- **gpt4.5_model.py**: GPT-4.5 with enhanced MoE and specialized reasoning experts
- **gpt5_model.py**: Theoretical GPT-5 with advanced MoE and dynamic context handling

## Architecture Differences

| Version | Parameters | Key Architecture Features                                  |
|---------|------------|------------------------------------------------------------|
| GPT-2   | 124M-1.5B  | Standard transformer with absolute position embeddings     |
| GPT-3   | 175B       | Scaled transformer with improved attention                 |
| GPT-3.5 | ~350B      | Optimized for instruction following                        |
| GPT-4   | ~1.7T      | Mixture of Experts (MoE) with specialized routing          |
| GPT-4.5 | ~3T        | Enhanced MoE with task-specific experts                    |
| GPT-5   | 1,000B     | Advanced MoE with dynamic context and unlimited attention  |

## Usage

Models are loaded through the version manager:

```python
from utils.version_manager import create_model

# Create a specific GPT version
model = create_model('gpt4')

# Override specific configuration parameters
model = create_model('gpt3.5', block_size=32768, n_head=20)
```
