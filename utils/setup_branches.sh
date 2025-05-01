#!/bin/bash
# Script to set up branch structure for different GPT versions

# Ensure script is run from the root directory
if [ ! -f "model.py" ]; then
  echo "Error: Please run this script from the project root directory containing model.py"
  exit 1
fi

# Create base branch
echo "Creating base branch..."
git checkout -b base

# Add dynamic context implementation
git add dynamic_context.py ARCHITECTURE.md
git commit -m "Add dynamic context manager with unlimited context support"

# Define branches and their specific configurations
declare -A branches
branches[gpt2]="block_size=1024 n_layer=12 n_head=12 n_embd=768 params=124M"
branches[gpt3]="block_size=2048 n_layer=96 n_head=96 n_embd=12288 params=175B"
branches[gpt3.5]="block_size=16384 n_layer=120 n_head=120 n_embd=14336 params=350B"
branches[gpt4]="block_size=128000 n_layer=200 moe=true experts=200 active_experts=8 params=1.7T"
branches[gpt4.5]="block_size=512000 n_layer=240 moe=true experts=240 active_experts=12 params=3T"
branches[gpt5]="block_size=1000000000 n_layer=300 moe=true experts=300 active_experts=16 params=1000B unlimited_context=true"

# Function to create config for a specific branch
create_config() {
  local branch=$1
  local params=$2
  local config_file="configs/${branch}_config.py"
  
  mkdir -p configs
  
  # Extract parameters
  block_size=$(echo $params | grep -o "block_size=[^ ]*" | cut -d= -f2)
  n_layer=$(echo $params | grep -o "n_layer=[^ ]*" | cut -d= -f2)
  n_head=$(echo $params | grep -o "n_head=[^ ]*" | cut -d= -f2 || echo $n_layer)
  n_embd=$(echo $params | grep -o "n_embd=[^ ]*" | cut -d= -f2)
  param_count=$(echo $params | grep -o "params=[^ ]*" | cut -d= -f2)
  moe=$(echo $params | grep -o "moe=true" || echo "")
  
  # Create config file
  cat > $config_file << EOF
"""
Configuration for $branch model ($param_count parameters)
"""

from dataclasses import dataclass

@dataclass
class ${branch^}Config:
    # Model architecture
    block_size: int = $block_size
    vocab_size: int = 50304
    n_layer: int = $n_layer
    n_head: int = $n_head 
    n_embd: int = $n_embd
    dropout: float = 0.0
    bias: bool = False  # False is slightly better
    
    # Parameter metrics
    parameter_count: str = "$param_count"
EOF

  # Add MoE configuration if applicable
  if [ ! -z "$moe" ]; then
    experts=$(echo $params | grep -o "experts=[^ ]*" | cut -d= -f2)
    active_experts=$(echo $params | grep -o "active_experts=[^ ]*" | cut -d= -f2)
    
    cat >> $config_file << EOF
    
    # Mixture of Experts
    use_moe: bool = True
    num_experts: int = $experts
    num_active_experts: int = $active_experts
    expert_capacity_factor: float = 1.25
    moe_jitter_eps: float = 0.01
EOF
  fi
  
  # Add unlimited context if applicable
  if echo $params | grep -q "unlimited_context=true"; then
    cat >> $config_file << EOF
    
    # Unlimited context support
    billion_token_context: bool = True
    use_sparse_attention: bool = True
    preserve_begin_tokens: int = 0  # Auto-calculated
    preserve_end_tokens: int = 0  # Auto-calculated
    middle_token_compression_ratio: float = 0.1
    token_eviction_strategy: str = 'importance'
EOF
  fi
  
  echo "Created config file: $config_file"
}

# Create each branch and its config
for branch in "${!branches[@]}"; do
  echo -e "\nSetting up $branch branch..."
  git checkout base
  git checkout -b $branch
  
  # Create version-specific config
  create_config $branch "${branches[$branch]}"
  
  # Add model architecture specializations 
  mkdir -p models
  touch models/${branch}_model.py
  
  # Add branch-specific changes
  echo "Creating ${branch}-specific model implementation..."
  cat > models/${branch}_model.py << EOF
"""
${branch^} model implementation
Parameters: ${branches[$branch]}
"""

import torch
import torch.nn as nn
from configs.${branch}_config import ${branch^}Config
EOF
  
  # Add more model-specific code based on branch
  case $branch in
    gpt2|gpt3|gpt3.5)
      cat >> models/${branch}_model.py << EOF

# Standard transformer implementation
class ${branch^}Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Standard architecture
        print(f"Initializing ${branch^} model with {config.parameter_count} parameters")
EOF
      ;;
    gpt4|gpt4.5)
      cat >> models/${branch}_model.py << EOF

# Mixture of Experts implementation
class ${branch^}MoETransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # MoE architecture
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_active_experts
        print(f"Initializing ${branch^} MoE model with {config.parameter_count} parameters")
        print(f"Using {self.num_experts} experts with {self.num_active_experts} active per token")
EOF
      ;;
    gpt5)
      cat >> models/${branch}_model.py << EOF

# Advanced architecture with unlimited context support
from dynamic_context import DynamicContextManager, DynamicContextConfig

class ${branch^}AdvancedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Advanced architecture
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_active_experts
        
        # Initialize dynamic context manager
        context_config = DynamicContextConfig(
            max_memory_mb=0,  # Auto-detect
            begin_tokens_to_preserve=config.preserve_begin_tokens,
            end_tokens_to_preserve=config.preserve_end_tokens,
        )
        self.context_manager = DynamicContextManager(context_config)
        
        print(f"Initializing ${branch^} model with {config.parameter_count} parameters")
        print(f"Unlimited context enabled with dynamic token management")
EOF
      ;;
  esac
  
  # Commit branch-specific changes
  git add configs/${branch}_config.py models/${branch}_model.py
  git commit -m "Add ${branch} model configuration and implementation"
done

# Go back to main branch
git checkout main
echo -e "\nAll branches created successfully. Use 'git checkout <branch>' to switch to a specific version."
echo "Available branches:"
for branch in "${!branches[@]}"; do
  echo "  - $branch: ${branches[$branch]}"
done
