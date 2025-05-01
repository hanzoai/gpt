# Demos

This directory contains demonstration scripts that showcase the features of the GPT implementation.

## Available Demos

- **simple_demo.py**: Simple demonstration of unlimited context with middle token eviction
- **unlimited_context_demo.py**: Comprehensive demo of the dynamic context management system
- **version_demo.py**: Demonstration of switching between different GPT versions

## Running the Demos

### Simple Demo

A basic demonstration of unlimited context with middle token eviction:

```bash
python demos/simple_demo.py
```

This demo shows how the system:
- Preserves beginning and end tokens
- Evicts middle tokens when memory constraints are reached
- Maintains context coherence despite token eviction

### Unlimited Context Demo

A more comprehensive demonstration of the dynamic context system:

```bash
python demos/unlimited_context_demo.py
```

Features demonstrated:
- Token importance scoring
- Memory monitoring and management
- Dynamic token eviction strategies
- Preservation of semantic structure

### Version Demo

Demonstration of loading and using different GPT versions:

```bash
python demos/version_demo.py
```

This shows:
- Configuration differences between versions
- Model parameter scaling
- Special features of each version
- Unlimited context configuration for GPT-5
