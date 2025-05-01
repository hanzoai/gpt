# Documentation

This directory contains detailed documentation for the GPT implementation.

## Available Documentation

- **LLM.md**: Comprehensive technical overview of the GPT architecture and upgrade paths
- **UNLIMITED_CONTEXT.md**: Detailed explanation of the unlimited context implementation
- **ARCHITECTURE.md**: Overview of the project architecture and branch structure

## Core Concepts

### Unlimited Context

The unlimited context implementation allows GPT models to process theoretically unlimited context lengths by:

1. Intelligently managing memory usage
2. Preserving beginning and end tokens
3. Evicting middle tokens based on importance
4. Dynamically scaling resources based on actual context size

See [UNLIMITED_CONTEXT.md](UNLIMITED_CONTEXT.md) for a complete explanation.

### Multiple GPT Versions

The project supports multiple GPT versions with different capabilities:

- **GPT-2**: Base implementation with standard transformer architecture
- **GPT-3/3.5**: Scaled transformers with improved attention mechanisms
- **GPT-4/4.5**: Mixture of Experts architectures with specialized routing
- **GPT-5**: Theoretical model with advanced MoE and dynamic context handling

See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details.

### LLM Architecture Overview

For a comprehensive overview of the architecture and upgrade paths, see [LLM.md](LLM.md), which covers:

- Current architecture analysis
- Step-by-step upgrade plans
- Implementation considerations
- Performance optimizations
- Memory management strategies
