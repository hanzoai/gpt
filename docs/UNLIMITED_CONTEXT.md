# Unlimited Context Support for GPT

This document explains how to use the unlimited context feature across different GPT versions.

## Overview

The unlimited context implementation allows GPT models to process theoretically unlimited context lengths while efficiently managing memory usage. Key features:

1. **Dynamic Memory Usage**: Only uses resources proportional to active context
2. **Intelligent Token Eviction**: Preserves beginning and end tokens while evicting middle tokens when memory is constrained
3. **Configurable Preservation**: Control what percentage of tokens to preserve in beginning vs. end
4. **Seamless Integration**: Works with all GPT versions with minimal configuration

## Enabling Unlimited Context

### Basic Usage

```python
from version_manager import load_model_config, create_model, enable_unlimited_context

# Load any GPT version (gpt2, gpt3, gpt3.5, gpt4, gpt4.5, gpt5)
model = create_model('gpt4')

# Enable unlimited context with custom preservation ratios
model = enable_unlimited_context(
    model,
    preserve_begin_ratio=0.1,  # Keep 10% of tokens from beginning
    preserve_end_ratio=0.2,    # Keep 20% of tokens from end
    max_memory_mb=0            # 0 = auto-detect available memory
)

# Model now supports unlimited context length
# When memory is constrained, it will automatically evict middle tokens
```

### Advanced Configuration

For more control over token preservation and eviction strategies:

```python
from dynamic_context import DynamicContextConfig, DynamicContextManager

# Create custom configuration
config = DynamicContextConfig(
    max_memory_mb=8192,                 # Memory limit in MB (0 = auto-detect)
    check_memory_every_n_tokens=1000,   # How often to check memory usage
    begin_tokens_to_preserve=2000,      # Fixed number of tokens to preserve at beginning
    end_tokens_to_preserve=5000,        # Fixed number of tokens to preserve at end
    token_eviction_strategy='importance', # Strategy: 'importance', 'age', 'hybrid'
    recency_weight=1.0,                 # Importance weight for token recency
    attention_weight=2.0,               # Importance weight for attention patterns
    content_weight=1.5                  # Importance weight for content features
)

# Create context manager with custom config
context_manager = DynamicContextManager(config)

# Attach to model
model.context_manager = context_manager
model.config.billion_token_context = True
```

## GPT Version Compatibility

| Version | Support Level | Notes |
|---------|--------------|-------|
| GPT-2   | Basic        | Limited by position embedding max length (1024) |
| GPT-3   | Good         | Requires extended position embeddings |
| GPT-3.5 | Better       | Native long context support (16K tokens) |
| GPT-4   | Excellent    | Optimized for long contexts (128K tokens) |
| GPT-4.5 | Superior     | Full support for unlimited context |
| GPT-5   | Optimal      | Designed specifically for efficient unlimited context |

## Implementation Details

The unlimited context system works by:

1. Monitoring memory usage during token processing
2. When memory constraints are detected:
   - Preserving a configurable number of tokens at the beginning
   - Preserving a configurable number of tokens at the end
   - Scoring middle tokens by importance (based on attention patterns, recency, content)
   - Evicting least important middle tokens first
   - Maintaining semantic "landmark" tokens in middle regions

During inference, the system seamlessly handles:
- Attention computation across preserved tokens
- Proper attention masking for causal relationships
- Correct position embedding for tokens after eviction

## Memory Usage Considerations

Memory usage scales primarily with the number of *active* tokens, not the total context length. Examples:

- 1B token context with 100K active tokens: ~similar memory to 100K context model
- 10K token usage in a 1B-capable model: minimal overhead compared to fixed context
- Evicted tokens: negligible memory impact (token IDs only, no activations/gradients)

## Best Practices

1. **Set Preservation Ratios Appropriately**: 
   - For conversations: preserve more at the end (recent messages)
   - For documents: balanced preservation or more at beginning (headers, context)

2. **Monitor Eviction Patterns**:
   - Log `context_manager.eviction_events` to understand when/where evictions occur
   - Adjust preservation ratios based on observed patterns

3. **Version-Specific Optimization**:
   - GPT-2/3: Use smaller contexts when possible
   - GPT-4/4.5/5: Take full advantage of unlimited context

## Example Applications

1. **Document Analysis**: Process entire books or long documents without chunking
2. **Extended Conversations**: Maintain full conversation history without truncation
3. **Code Understanding**: Parse and comprehend entire codebases with proper context
4. **Research Analysis**: Process multiple research papers together with cross-references

## Limitations

1. **Middle Context Gaps**: Evicted middle tokens create discontinuities in context
2. **Memory Trade-offs**: Higher preservation ratios require more memory
3. **Attention Computation**: Large active contexts still require significant computation
4. **Position Embedding Limits**: Model-dependent limitations on absolute positions

For more information, see the full implementation in `dynamic_context.py` and examples in `unlimited_context_demo.py`.
