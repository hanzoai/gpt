# Utilities

This directory contains utility modules and helper scripts for the GPT implementation.

## Contents

- **sparse_attention.py**: Implementation of sparse attention mechanisms for billion-token context support
- **dynamic_context.py**: Dynamic context management with middle token eviction for unlimited context length
- **version_manager.py**: Utilities for managing different GPT versions and configurations
- **setup_branches.sh**: Script to set up the branch structure for different GPT versions

## Usage

### Version Manager

```python
from utils.version_manager import create_model, enable_unlimited_context

# Create a specific GPT version
model = create_model('gpt4')

# Enable unlimited context
model = enable_unlimited_context(model, preserve_begin_ratio=0.1, preserve_end_ratio=0.2)
```

### Dynamic Context Management

```python
from utils.dynamic_context import DynamicContextManager, DynamicContextConfig

# Create a custom configuration
config = DynamicContextConfig(
    max_memory_mb=8192,
    begin_tokens_to_preserve=1000,
    end_tokens_to_preserve=2000
)

# Initialize the context manager
context_manager = DynamicContextManager(config)
```

### Branch Setup

To set up the branch structure for different GPT versions:

```bash
cd utils
./setup_branches.sh
```
