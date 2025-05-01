"""
Dynamic Memory Manager for unlimited context length.
Handles token lifecycle with dynamic eviction strategies optimized for
preserving beginning/end tokens while efficiently managing middle regions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class TokenImportanceScorer:
    """Scores token importance for eviction decisions."""
    
    def __init__(self, config):
        self.config = config
        self.attention_weights_history = []
        self.token_usage_counts = {}
        self.token_age = {}
        
    def score_tokens(self, 
                     tokens: torch.Tensor, 
                     position_ids: torch.Tensor, 
                     attention_weights: Optional[torch.Tensor] = None
                    ) -> torch.Tensor:
        """
        Score tokens based on multiple factors:
        - Position (beginning/end are more important)
        - Attention received in previous layers
        - Token rarity/uniqueness
        - Recency of use
        
        Args:
            tokens: Input token ids (batch_size, seq_len)
            position_ids: Position ids for tokens
            attention_weights: Optional attention weights from previous layers
            
        Returns:
            Importance scores for each token (batch_size, seq_len)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Start with position-based scores (beginning/end are high priority)
        seq_positions = torch.arange(seq_len, device=device).float()
        
        # Favor beginning and end positions
        begin_end_preference = torch.ones(seq_len, device=device)
        begin_percentage = self.config.preserve_begin_percent
        end_percentage = self.config.preserve_end_percent
        
        # Mark beginning tokens as high importance
        begin_count = max(1, int(seq_len * begin_percentage))
        begin_end_preference[:begin_count] = 10.0
        
        # Mark ending tokens as high importance
        end_count = max(1, int(seq_len * end_percentage))
        begin_end_preference[-end_count:] = 10.0
        
        # Middle positions have varying importance
        middle_start = begin_count
        middle_end = seq_len - end_count
        if middle_end > middle_start:
            # Create a V-shaped importance curve for middle tokens
            middle_len = middle_end - middle_start
            middle_positions = torch.arange(middle_len, device=device).float()
            
            # V-curve calculation (lowest in the middle, higher at edges)
            normalized_positions = middle_positions / (middle_len - 1)
            v_curve = 1.0 - 2.0 * torch.abs(normalized_positions - 0.5)
            
            # Apply the V-curve to middle positions
            begin_end_preference[middle_start:middle_end] = 1.0 + v_curve * 2.0
        
        # If we have attention weights, use them to adjust importance
        attention_based_importance = torch.ones(seq_len, device=device)
        if attention_weights is not None:
            # Average attention received by each token
            attention_per_token = attention_weights.mean(dim=(0, 1))
            
            # Normalize to 0-2 range (1 is neutral)
            attention_based_importance = 1.0 + (attention_per_token - attention_per_token.mean()) / (attention_per_token.std() + 1e-5)
        
        # Combine factors for final score
        importance_scores = begin_end_preference * attention_based_importance
        
        # Expand to batch dimension
        return importance_scores.expand(batch_size, seq_len)

class TokenCompressor:
    """Compresses tokens for memory efficiency."""
    
    def __init__(self, config):
        self.config = config
        self.compression_levels = 3  # Number of compression tiers
        
    def compress_tokens(self, 
                       hidden_states: torch.Tensor, 
                       importance_scores: torch.Tensor, 
                       memory_threshold: float,
                       ) -> Dict:
        """
        Compress tokens based on importance and memory constraints.
        
        Args:
            hidden_states: Token hidden states (batch_size, seq_len, hidden_dim)
            importance_scores: Token importance scores (batch_size, seq_len)
            memory_threshold: Percentage of tokens to keep at full resolution
            
        Returns:
            Dictionary with compressed tokens and metadata
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Calculate how many tokens to keep at each compression level
        full_resolution_count = max(1, int(seq_len * memory_threshold))
        
        # Sort tokens by importance
        _, sorted_indices = torch.sort(importance_scores, dim=1, descending=True)
        
        # Identify tokens for each compression tier
        tier1_indices = sorted_indices[:, :full_resolution_count]  # Full resolution
        tier2_indices = sorted_indices[:, full_resolution_count:full_resolution_count*2]  # Medium compression
        tier3_indices = sorted_indices[:, full_resolution_count*2:]  # High compression
        
        # Create compression masks
        tier1_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        tier2_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        tier3_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Fill in masks per batch
        for b in range(batch_size):
            tier1_mask[b, tier1_indices[b]] = True
            tier2_mask[b, tier2_indices[b]] = True
            tier3_mask[b, tier3_indices[b]] = True
        
        # Perform compression
        # Tier 1: Full resolution (no compression)
        tier1_tokens = hidden_states.clone()
        
        # Tier 2: Medium compression (50% dim reduction)
        tier2_compression = nn.Linear(hidden_dim, hidden_dim // 2, device=device)
        tier2_tokens = hidden_states.clone()
        tier2_tokens[tier2_mask] = self._expand_compressed(
            tier2_compression(hidden_states[tier2_mask]), hidden_dim
        )
        
        # Tier 3: High compression (90% dim reduction)
        tier3_compression = nn.Linear(hidden_dim, hidden_dim // 10, device=device)
        tier3_tokens = hidden_states.clone()
        tier3_tokens[tier3_mask] = self._expand_compressed(
            tier3_compression(hidden_states[tier3_mask]), hidden_dim
        )
        
        # Combine according to masks
        combined = torch.zeros_like(hidden_states)
        combined[tier1_mask] = tier1_tokens[tier1_mask]
        combined[tier2_mask] = tier2_tokens[tier2_mask]
        combined[tier3_mask] = tier3_tokens[tier3_mask]
        
        # Return compressed tokens and metadata
        return {
            'hidden_states': combined,
            'tier1_indices': tier1_indices,
            'tier2_indices': tier2_indices,
            'tier3_indices': tier3_indices,
            'tier1_mask': tier1_mask,
            'tier2_mask': tier2_mask,
            'tier3_mask': tier3_mask,
            'compression_ratio': 1.0 - (full_resolution_count / seq_len)
        }
    
    def _expand_compressed(self, compressed: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Expand compressed representations back to original dimension with padding."""
        orig_shape = compressed.shape
        
        # Simple padding approach
        padding_size = target_dim - compressed.shape[-1]
        padded = torch.nn.functional.pad(compressed, (0, padding_size))
        
        return padded

class DynamicMemoryManager:
    """
    Manages token lifecycle for unlimited context.
    Handles eviction, compression, and retrieval.
    """
    
    def __init__(self, config):
        self.config = config
        self.max_active_tokens = config.max_active_tokens
        self.scorer = TokenImportanceScorer(config)
        self.compressor = TokenCompressor(config)
        self.token_history = {}
        self.eviction_history = []
        
    def manage_context(self, 
                      hidden_states: torch.Tensor, 
                      tokens: torch.Tensor,
                      position_ids: torch.Tensor,
                      attention_weights: Optional[torch.Tensor] = None
                     ) -> Dict:
        """
        Main entry point for memory management.
        Handles token importance scoring, compression, and eviction.
        
        Args:
            hidden_states: Token hidden states (batch_size, seq_len, hidden_dim)
            tokens: Input token ids (batch_size, seq_len)
            position_ids: Position ids for tokens
            attention_weights: Optional attention weights
            
        Returns:
            Dictionary with processed tokens and memory management metadata
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # If context length is small, no memory management needed
        if seq_len <= self.max_active_tokens:
            return {'hidden_states': hidden_states, 'memory_managed': False}
        
        # Score tokens for importance
        importance_scores = self.scorer.score_tokens(
            tokens, position_ids, attention_weights
        )
        
        # Calculate memory threshold based on config
        memory_ratio = self.max_active_tokens / seq_len
        memory_threshold = max(0.1, memory_ratio)  # At least 10% of tokens at full res
        
        # Compress tokens based on importance
        compression_result = self.compressor.compress_tokens(
            hidden_states, importance_scores, memory_threshold
        )
        
        # Update token history (for retrieval)
        self._update_token_history(tokens, hidden_states, position_ids)
        
        # Log memory management stats
        stats = {
            'total_tokens': seq_len,
            'active_tokens': self.max_active_tokens,
            'memory_ratio': memory_ratio,
            'compression_ratio': compression_result['compression_ratio']
        }
        
        return {
            'hidden_states': compression_result['hidden_states'],
            'memory_managed': True,
            'stats': stats,
            'tier1_mask': compression_result['tier1_mask'],
            'tier2_mask': compression_result['tier2_mask'],
            'tier3_mask': compression_result['tier3_mask']
        }
    
    def _update_token_history(self, 
                             tokens: torch.Tensor, 
                             hidden_states: torch.Tensor,
                             position_ids: torch.Tensor
                            ) -> None:
        """Update token history for potential retrieval."""
        # In a real implementation, this would maintain a history of tokens
        # for retrieval when needed. Here we just keep track of the latest state.
        batch_idx = 0  # For simplicity, track only the first batch item
        
        for pos, (token, state) in enumerate(zip(tokens[batch_idx], hidden_states[batch_idx])):
            token_id = token.item()
            self.token_history[pos] = {
                'token_id': token_id,
                'hidden_state': state.detach().cpu(),  # Store on CPU to save GPU memory
                'position_id': position_ids[batch_idx, pos].item(),
                'last_access': 0  # Will be updated on access
            }
    
    def retrieve_token(self, position: int) -> Optional[Dict]:
        """Retrieve a token from history by position."""
        if position in self.token_history:
            token_data = self.token_history[position]
            # Update access counter
            token_data['last_access'] += 1
            return token_data
        return None

# Configuration class for dynamic memory
class DynamicMemoryConfig:
    def __init__(self, 
                max_active_tokens: int = 16384,
                preserve_begin_percent: float = 0.1,
                preserve_end_percent: float = 0.2,
                compression_levels: int = 3,
                use_attention_weights: bool = True,
                eviction_strategy: str = 'importance'):
        self.max_active_tokens = max_active_tokens
        self.preserve_begin_percent = preserve_begin_percent
        self.preserve_end_percent = preserve_end_percent
        self.compression_levels = compression_levels
        self.use_attention_weights = use_attention_weights
        self.eviction_strategy = eviction_strategy

# Initialize for a GPT model
def init_dynamic_memory(model, max_active_tokens=None):
    """
    Initialize dynamic memory management for a GPT model.
    
    Args:
        model: GPT model instance
        max_active_tokens: Maximum tokens to keep active in memory
        
    Returns:
        Model with dynamic memory management
    """
    # Use model's block size as default if not specified
    if max_active_tokens is None:
        max_active_tokens = getattr(model.config, 'block_size', 16384)
    
    # Create memory config
    memory_config = DynamicMemoryConfig(
        max_active_tokens=max_active_tokens,
        preserve_begin_percent=0.1,
        preserve_end_percent=0.2,
        compression_levels=3,
        use_attention_weights=True,
        eviction_strategy='importance'
    )
    
    # Attach memory manager to model
    model.memory_manager = DynamicMemoryManager(memory_config)
    
    # Patch model's forward method to use memory management
    original_forward = model.forward
    
    def forward_with_memory_management(self, idx, targets=None, *args, **kwargs):
        # Extract relevant info for memory management
        device = idx.device
        batch_size, seq_len = idx.shape
        
        # Generate position IDs
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(idx)
        
        # Check if memory management is needed
        if hasattr(self, 'memory_manager') and seq_len > self.memory_manager.max_active_tokens:
            # Get embeddings first
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(position_ids.reshape(-1))
            pos_emb = pos_emb.reshape(batch_size, seq_len, -1)
            x = self.transformer.drop(tok_emb + pos_emb)
            
            # Apply memory management
            memory_result = self.memory_manager.manage_context(
                x, idx, position_ids
            )
            
            # Use managed hidden states
            x = memory_result['hidden_states']
            
            # Log stats if memory was managed
            if memory_result['memory_managed']:
                print(f"Memory managed: {memory_result['stats']}")
                
            # Continue with normal forward pass (skipping embedding)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            
            # Rest of the forward pass
            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                      targets.view(-1), ignore_index=-1)
                return logits, loss
            else:
                logits = self.lm_head(x[:, [-1], :])
                return logits, None
        else:
            # Default forward pass
            return original_forward(idx, targets, *args, **kwargs)
    
    # Replace forward method
    model.forward = types.MethodType(forward_with_memory_management, model)
    
    return model
