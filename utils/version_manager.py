#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT Version Manager - Load and configure different model versions.
This module provides utilities to easily switch between GPT2, GPT3, GPT3.5, GPT4, GPT4.5
and the theoretical GPT5 configurations.
"""

import os
import importlib
import sys
import json

# Dictionary of model version sizes and defaults
MODEL_VERSIONS = {
    'gpt2': {
        'params': '124M',
        'block_size': 1024,
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768
    },
    'gpt3': {
        'params': '175B',
        'block_size': 2048,
        'n_layer': 96,
        'n_head': 96,
        'n_embd': 12288
    },
    'gpt3_5': {
        'params': '350B',
        'block_size': 16384,
        'n_layer': 120,
        'n_head': 120,
        'n_embd': 14336
    },
    'gpt4': {
        'params': '1.7T',
        'block_size': 128000,
        'n_layer': 200,
        'n_head': 200,
        'n_embd': 18432,
        'use_moe': True,
        'num_experts': 200,
        'num_active_experts': 8
    },
    'gpt4_5': {
        'params': '3T',
        'block_size': 512000,
        'n_layer': 240,
        'n_head': 240,
        'n_embd': 20480,
        'use_moe': True,
        'num_experts': 240,
        'num_active_experts': 12
    },
    'gpt5': {
        'params': '1000B',
        'block_size': 1000000000,  # 1B tokens (theoretical)
        'n_layer': 300,
        'n_head': 300,
        'n_embd': 22528,
        'use_moe': True,
        'num_experts': 300,
        'num_active_experts': 16,
        'billion_token_context': True,
        'use_sparse_attention': True
    }
}

def load_dynamic_context_if_available():
    """Load the dynamic context manager if available."""
    try:
        sys.path.append(os.getcwd())
        import dynamic_context
        return (dynamic_context.DynamicContextManager, 
                dynamic_context.DynamicContextConfig)
    except ImportError:
        print("Warning: dynamic_context module not found. Unlimited context features disabled.")
        return None, None

def load_model_config(version):
    """
    Load configuration for the specified model version.
    
    Args:
        version: Model version ('gpt2', 'gpt3', 'gpt3.5', 'gpt4', 'gpt4.5', 'gpt5')
        
    Returns:
        Configuration object for the specified version
    """
    if version not in MODEL_VERSIONS:
        valid_versions = ", ".join(MODEL_VERSIONS.keys())
        raise ValueError("Unknown model version: %s. Valid versions: %s" % (version, valid_versions))
    
    # Check if version-specific config module exists
    config_path = os.path.join('configs', version + "_config.py")
    
    if os.path.exists(config_path):
        try:
            # Import the module
            sys.path.append(os.path.dirname(config_path))
            config_module_name = version + "_config"
            config_module = __import__(config_module_name)
            
            # Get the config class (assumes class name format is VersionConfig, e.g. Gpt2Config)
            class_name = version.replace('.', '_').capitalize() + "Config"
            if hasattr(config_module, class_name):
                config_class = getattr(config_module, class_name)
                return config_class()
        except Exception as e:
            print("Error loading config from file: %s" % str(e))
    
    # If config module doesn't exist or class not found, use built-in defaults
    from model import GPTConfig
    params = MODEL_VERSIONS[version]
    
    # Create GPTConfig with version-specific parameters
    config = GPTConfig(
        block_size=params.get('block_size', 1024),
        vocab_size=50304,  # Standard for all versions
        n_layer=params.get('n_layer', 12),
        n_head=params.get('n_head', 12),
        n_embd=params.get('n_embd', 768),
        dropout=params.get('dropout', 0.0),
        bias=params.get('bias', False)
    )
    
    # Add MoE parameters if applicable
    if params.get('use_moe', False):
        config.use_moe = True
        config.num_experts = params.get('num_experts', 8)
        config.num_active_experts = params.get('num_active_experts', 2)
    
    # Add unlimited context parameters if applicable
    if params.get('billion_token_context', False):
        config.billion_token_context = True
        config.use_sparse_attention = params.get('use_sparse_attention', True)
        
        # Dynamic context handling
        DynamicContextManager, DynamicContextConfig = load_dynamic_context_if_available()
        if DynamicContextManager is not None:
            config.dynamic_context_available = True
    
    return config

def create_model(version, **override_args):
    """
    Create a model of the specified version with optional parameter overrides.
    
    Args:
        version: Model version ('gpt2', 'gpt3', etc.)
        **override_args: Optional parameter overrides
        
    Returns:
        Configured model instance
    """
    # Load configuration
    config = load_model_config(version)
    
    # Apply overrides
    for key, value in override_args.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print("Warning: Config has no attribute '%s', skipping override" % key)
    
    # Check if version has a custom model implementation
    model_path = os.path.join('models', version + "_model.py")
    if os.path.exists(model_path):
        try:
            # Import the model module
            sys.path.append(os.path.dirname(model_path))
            model_module_name = version + "_model"
            model_module = __import__(model_module_name)
            
            # Get the model class based on version and architecture
            if getattr(config, 'use_moe', False) and getattr(config, 'billion_token_context', False):
                class_name = version.replace('.', '_').capitalize() + "AdvancedTransformer"
            elif getattr(config, 'use_moe', False):
                class_name = version.replace('.', '_').capitalize() + "MoETransformer"
            else:
                class_name = version.replace('.', '_').capitalize() + "Transformer"
            
            if hasattr(model_module, class_name):
                model_class = getattr(model_module, class_name)
                return model_class(config)
        except Exception as e:
            print("Error loading model implementation: %s" % str(e))
    
    # Fall back to standard GPT implementation
    from model import GPT
    return GPT(config)

def enable_unlimited_context(model, preserve_begin_ratio=0.1, preserve_end_ratio=0.2, max_memory_mb=0):
    """
    Enable unlimited context features on a model.
    
    Args:
        model: The model to modify
        preserve_begin_ratio: Ratio of tokens to preserve at beginning
        preserve_end_ratio: Ratio of tokens to preserve at end
        max_memory_mb: Maximum memory to use (0 = auto-detect)
        
    Returns:
        Modified model with unlimited context
    """
    # Check if model supports unlimited context
    if not hasattr(model.config, 'billion_token_context'):
        print("Warning: Adding billion_token_context attribute to model config")
        model.config.billion_token_context = True
    
    # Initialize model with dynamic context
    try:
        sys.path.append(os.getcwd())
        from dynamic_context import create_context_manager
        
        # Create and attach context manager
        context_manager = create_context_manager(
            max_memory_mb=max_memory_mb,
            begin_ratio=preserve_begin_ratio,
            end_ratio=preserve_end_ratio
        )
        
        # Attach to model
        model.context_manager = context_manager
        print("Unlimited context enabled with: begin=%.2f, end=%.2f" % 
              (preserve_begin_ratio, preserve_end_ratio))
        
        # Modify config
        model.config.preserve_begin_tokens = 0  # Use auto-calculated
        model.config.preserve_end_tokens = 0  # Use auto-calculated
        model.config.use_sparse_attention = True
        
        return model
    except ImportError:
        print("Error: dynamic_context module required for unlimited context")
        return model

def print_model_comparison():
    """Print a comparison of all model versions."""
    print("\n" + "=" * 100)
    fmt = "%-12s %-10s %-12s %-8s %-8s %-8s %-5s %-10s"
    print(fmt % ("Model Version", "Parameters", "Block Size", "Layers", "Heads", "Embd", "MoE", "Unlimited"))
    print("-" * 100)
    
    for version, params in MODEL_VERSIONS.items():
        moe_flag = "YES" if params.get('use_moe', False) else "NO"
        unlimited_flag = "YES" if params.get('billion_token_context', False) else "NO"
        
        print(fmt % (
            version, 
            params['params'], 
            str(params['block_size']), 
            str(params['n_layer']), 
            str(params['n_head']), 
            str(params['n_embd']),
            moe_flag,
            unlimited_flag
        ))
    
    print("=" * 100)

def example_usage():
    """Show example usage of the version manager."""
    print("\nExample Usage:")
    print("1. Load a specific model version:")
    print("   model = version_manager.create_model('gpt3_5')")
    print("\n2. Enable unlimited context on any model:")
    print("   model = version_manager.enable_unlimited_context(model, preserve_begin_ratio=0.1, preserve_end_ratio=0.2)")
    print("\n3. Override configuration parameters:")
    print("   model = version_manager.create_model('gpt4', block_size=65536, dropout=0.1)")

if __name__ == "__main__":
    # Print model comparison when run directly
    print_model_comparison()
    example_usage()
    
    # Example of loading a specific version
    print("\nExample: Loading GPT-5 configuration...")
    try:
        config = load_model_config('gpt5')
        
        # Print config details in a Python 2.7 compatible way
        config_dict = vars(config)
        filtered_dict = {}
        for k, v in config_dict.items():
            if not k.startswith('_'):
                filtered_dict[k] = v
        
        print(json.dumps(filtered_dict, indent=2))
    except Exception as e:
        print("Error loading GPT-5 configuration: %s" % str(e))
