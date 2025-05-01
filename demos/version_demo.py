"""
Demonstration of version switching between different GPT models.
Shows how to configure and use GPT-2, GPT-3, GPT-3.5, GPT-4, GPT-4.5 and GPT-5.
"""

import os
import sys
import time
from version_manager import print_model_comparison, load_model_config, create_model

def print_banner(text):
    """Print a banner with text."""
    width = len(text) + 4
    print("=" * width)
    print("= " + text + " =")
    print("=" * width)

def print_info(text):
    """Print information text."""
    print(">>> " + text)

def demonstrate_versions():
    """Demonstrate switching between different GPT versions."""
    print_banner("GPT VERSION COMPARISON")
    print_model_comparison()
    
    # Demonstrate loading configurations for different versions
    print_banner("MODEL CONFIGURATIONS")
    
    for version in ['gpt2', 'gpt3', 'gpt3.5', 'gpt4', 'gpt4.5', 'gpt5']:
        print_info("Loading " + version + " configuration...")
        
        try:
            config = load_model_config(version)
            
            # Print key configuration parameters
            print("Version:       " + version)
            print("Params:        " + MODEL_VERSIONS[version]['params'])
            print("Layers:        " + str(config.n_layer))
            print("Heads:         " + str(config.n_head))
            print("Embedding:     " + str(config.n_embd))
            print("Context size:  " + str(config.block_size))
            print("MoE:           " + ("Yes" if hasattr(config, 'use_moe') and config.use_moe else "No"))
            print("Unlimited ctx: " + ("Yes" if hasattr(config, 'billion_token_context') and 
                                     config.billion_token_context else "No"))
            print("")
        except Exception as e:
            print("Error loading " + version + ": " + str(e))
            print("")
    
    # Demonstrate GPT5 with unlimited context
    print_banner("GPT-5 WITH UNLIMITED CONTEXT")
    
    print_info("Creating GPT-5 model with unlimited context...")
    try:
        # In a real implementation, this would actually load the model
        # Here we just create a placeholder for demonstration
        gpt5_config = load_model_config('gpt5')
        
        # Settings for unlimited context
        gpt5_config.billion_token_context = True
        gpt5_config.preserve_begin_tokens = int(1e9 * 0.1)  # 10% of 1B tokens
        gpt5_config.preserve_end_tokens = int(1e9 * 0.2)    # 20% of 1B tokens
        
        print("GPT-5 configuration:")
        print("- Billion token context enabled")
        print("- Will preserve " + str(gpt5_config.preserve_begin_tokens) + " beginning tokens")
        print("- Will preserve " + str(gpt5_config.preserve_end_tokens) + " end tokens")
        print("- Block size: " + str(gpt5_config.block_size) + " tokens")
        
        if hasattr(gpt5_config, 'use_moe') and gpt5_config.use_moe:
            print("- Using Mixture of Experts with " + str(gpt5_config.num_experts) + 
                  " total experts, " + str(gpt5_config.num_active_experts) + " active")
    
    except Exception as e:
        print("Error creating GPT-5 model: " + str(e))

# Define the model versions for demonstration
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
    'gpt3.5': {
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
    'gpt4.5': {
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
        'block_size': 1000000000,  # 1B tokens
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

if __name__ == "__main__":
    demonstrate_versions()
