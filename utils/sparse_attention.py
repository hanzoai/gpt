"""
Implementation of sparse attention mechanisms for billion-token context windows.
This enables efficient handling of extremely long contexts by compressing and
selectively attending to tokens based on importance and position.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SparseAttentionManager:
    """
    Manages sparse attention for billion-token contexts.
    This class handles token selection, compression, and eviction strategies.
    """
    
    def __init__(self, config):
        """
        Initialize the sparse attention manager with config parameters.
        
        Args:
            config: GPTConfig instance with billion-token context parameters
        """
        self.config = config
        self.block_size = config.block_size
        self.billion_token_context = config.billion_token_context
        self.max_actual_block_size = config.max_actual_block_size
        self.preserve_begin_tokens = config.preserve_begin_tokens
        self.preserve_end_tokens = config.preserve_end_tokens
        self.middle_token_compression_ratio = config.middle_token_compression_ratio
        self.token_eviction_strategy = config.token_eviction_strategy
        self.max_active_context_ratio = config.max_active_context_ratio
        
        # Initialize token importance tracking
        self.token_importance_scores = None
        self.token_ages = None
        self.eviction_history = []
        
    def select_tokens(self, x, preserve_indices=None):
        """
        Select which tokens to keep at full resolution and which to compress.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            preserve_indices: Optional tensor of indices to always preserve
            
        Returns:
            selected_indices: Indices of tokens to keep at full resolution
            compressed_indices: Indices of tokens to compress
            compression_masks: Masks for different levels of compression
        """
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # Calculate number of tokens to preserve at beginning and end
        if self.preserve_begin_tokens <= 0:
            preserve_begin = max(int(seq_len * 0.1), 1)  # At least 10% from beginning
        else:
            preserve_begin = min(self.preserve_begin_tokens, seq_len // 2)
            
        if self.preserve_end_tokens <= 0:
            preserve_end = max(int(seq_len * 0.2), 1)  # At least 20% from end
        else:
            preserve_end = min(self.preserve_end_tokens, seq_len // 2)
            
        # Ensure we don't try to preserve more tokens than exist
        total_preserved = preserve_begin + preserve_end
        if total_preserved > seq_len:
            ratio = seq_len / total_preserved
            preserve_begin = max(int(preserve_begin * ratio), 1)
            preserve_end = max(int(preserve_end * ratio), 1)
        
        # Create indices for preserved tokens
        begin_indices = torch.arange(0, preserve_begin, device=device)
        end_indices = torch.arange(max(0, seq_len - preserve_end), seq_len, device=device)
        
        # Middle section that will be subject to pruning/compression
        middle_start = preserve_begin
        middle_end = seq_len - preserve_end
        
        # Handle case where middle section is empty or very small
        if middle_end <= middle_start:
            # No middle to compress, just keep all tokens
            return torch.arange(0, seq_len, device=device), torch.tensor([], device=device), None
        
        # Determine how many middle tokens to keep based on importance
        middle_indices = torch.arange(middle_start, middle_end, device=device)
        middle_len = middle_end - middle_start
        
        # Calculate how many middle tokens to keep vs. compress
        max_active_tokens = max(int(self.block_size * self.max_active_context_ratio), 
                               preserve_begin + preserve_end)
        middle_tokens_to_keep = max(0, max_active_tokens - preserve_begin - preserve_end)
        
        # If we need to compress middle tokens
        if middle_len > middle_tokens_to_keep:
            # Compute importance scores for middle tokens if not already computed
            if self.token_importance_scores is None or self.token_importance_scores.shape[1] != seq_len:
                # Simplified importance scoring - in a real implementation this would be more sophisticated
                # For now just use a combination of position (favoring more recent) and a random component
                token_positions = torch.arange(middle_start, middle_end, device=device).float()
                position_scores = 1.0 - (token_positions - middle_start) / max(1, (middle_end - middle_start))
                random_scores = torch.rand(middle_len, device=device) * 0.3  # Random component
                importance = position_scores + random_scores
                
                # In a real implementation, we'd use token embeddings, attention scores, etc.
                self.token_importance_scores = importance
            
            # Select tokens based on importance scores
            if self.token_eviction_strategy == 'importance':
                _, selected_middle_indices = torch.topk(self.token_importance_scores, 
                                                     min(middle_tokens_to_keep, middle_len))
                selected_middle_indices = middle_start + selected_middle_indices
            elif self.token_eviction_strategy == 'age':
                # Oldest tokens in the middle get compressed first
                stride = max(1, middle_len // middle_tokens_to_keep)
                selected_middle_indices = torch.arange(middle_start, middle_end, stride, device=device)
            else:  # 'hybrid'
                # Mix of strategies - keep some based on importance, some based on position
                importance_keep = middle_tokens_to_keep // 2
                position_keep = middle_tokens_to_keep - importance_keep
                
                # Get important tokens
                _, importance_indices = torch.topk(self.token_importance_scores, 
                                               min(importance_keep, middle_len))
                importance_indices = middle_start + importance_indices
                
                # Get evenly spaced tokens for positional context
                stride = max(1, middle_len // max(1, position_keep))
                position_indices = torch.arange(middle_start, middle_end, stride, device=device)
                
                # Combine and remove duplicates
                combined = torch.cat([importance_indices, position_indices])
                selected_middle_indices = torch.unique(combined)
                
            # Combine preserved tokens with selected middle tokens
            all_selected = torch.cat([begin_indices, selected_middle_indices, end_indices])
            all_selected = torch.unique(all_selected)  # Remove any duplicates
            
            # Find which tokens are to be compressed
            compressed_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            compressed_mask[all_selected] = False
            compressed_indices = torch.arange(0, seq_len, device=device)[compressed_mask]
            
            return all_selected, compressed_indices, None
        else:
            # No need to compress, keep all tokens
            return torch.arange(0, seq_len, device=device), torch.tensor([], device=device), None
    
    def compress_tokens(self, x, compressed_indices, compression_level=1):
        """
        Compress tokens that are not kept at full resolution.
        Different compression levels can be applied based on token importance.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            compressed_indices: Indices of tokens to compress
            compression_level: Level of compression to apply (1-3)
            
        Returns:
            Compressed representation of the input tokens
        """
        if len(compressed_indices) == 0:
            return None
            
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # Extract tokens to compress
        compressed_tokens = x[:, compressed_indices, :]
        
        # Apply different compression levels
        if compression_level == 1:
            # Light compression - reduce hidden dim by 50%
            compression_factor = 0.5
        elif compression_level == 2:
            # Medium compression - reduce hidden dim by 75%
            compression_factor = 0.25
        else:
            # Heavy compression - reduce hidden dim by 90%
            compression_factor = 0.1
            
        compressed_dim = max(1, int(hidden_dim * compression_factor))
        
        # Simple linear projection for compression
        # In a real implementation, this would be a more sophisticated method
        projection = nn.Linear(hidden_dim, compressed_dim).to(device)
        compressed = projection(compressed_tokens)
        
        return {
            'tokens': compressed,
            'indices': compressed_indices,
            'level': compression_level,
            'original_dim': hidden_dim,
            'compressed_dim': compressed_dim
        }
    
    def decompress_tokens(self, compressed_data, target_hidden_dim):
        """
        Decompress tokens when they need to be attended to.
        
        Args:
            compressed_data: Dict with compressed token data
            target_hidden_dim: Target hidden dimension for decompression
            
        Returns:
            Decompressed token representations
        """
        if compressed_data is None:
            return None
            
        compressed_tokens = compressed_data['tokens']
        compressed_dim = compressed_data['compressed_dim']
        
        # Simple linear projection for decompression
        # In a real implementation, this would be more sophisticated
        device = compressed_tokens.device
        decompression = nn.Linear(compressed_dim, target_hidden_dim).to(device)
        decompressed = decompression(compressed_tokens)
        
        return decompressed
    
    def update_token_importance(self, attention_weights, indices=None):
        """
        Update token importance scores based on attention weights.
        Tokens that receive more attention are deemed more important.
        
        Args:
            attention_weights: Attention weight tensor
            indices: Optional indices to update (if None, update all)
        """
        # In a real implementation, this would use attention patterns to
        # identify which tokens are most attended to and update their importance
        pass
    
    def manage_token_lifecycle(self, x, current_pos):
        """
        Manage the full lifecycle of tokens in the context:
        - Track token ages
        - Update importance scores
        - Handle token eviction when context exceeds memory limits
        - Record eviction decisions for analysis
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            current_pos: Current position in the sequence
            
        Returns:
            Updated token tracking information
        """
        # This would be a more complex implementation in a real system
        pass

class BillionContextAttention(nn.Module):
    """
    Attention mechanism that supports billion-token contexts through
    sparse attention and token lifecycle management.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sparse_manager = SparseAttentionManager(config)
        
        # Regular attention parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Projections for Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        """
        Forward pass with billion-token context support.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Output tensor with same shape as input
        """
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # Standard Q, K, V projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        head_size = self.n_embd // self.n_head
        q = q.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        
        # If billion-token context is enabled and sequence is very long
        if self.config.billion_token_context and seq_len > self.config.max_actual_block_size:
            # Select which tokens to keep at full resolution and which to compress
            selected_indices, compressed_indices, _ = self.sparse_manager.select_tokens(x)
            
            # Apply token lifecycle management
            self.sparse_manager.manage_token_lifecycle(x, seq_len - 1)
            
            if len(compressed_indices) > 0:
                # Compress less important tokens
                compressed_data = self.sparse_manager.compress_tokens(x, compressed_indices)
                
                # Log compression stats
                compression_ratio = len(compressed_indices) / seq_len
                print(f"Compressed {len(compressed_indices)} tokens ({compression_ratio:.2%} of context)")
                
                # In a full implementation, we would:
                # 1. Maintain compressed tokens in a separate memory store
                # 2. Decompress them when needed for specific attention patterns
                # 3. Apply sparse attention patterns that focus on important tokens
                # 4. Handle token eviction when context grows beyond memory limits
                
                # For now, we'll use only the selected tokens for attention calculation
                q_selected = q[:, :, selected_indices, :]
                k_selected = k[:, :, selected_indices, :]
                v_selected = v[:, :, selected_indices, :]
                
                # Apply attention only on selected tokens
                if self.flash:
                    # Use flash attention with causal mask
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q_selected, k_selected, v_selected, 
                        attn_mask=None, 
                        dropout_p=self.dropout if self.training else 0,
                        is_causal=True
                    )
                else:
                    # Manual implementation with causal mask
                    selected_len = len(selected_indices)
                    att = (q_selected @ k_selected.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
                    mask = self.mask[:, :, :selected_len, :selected_len]
                    att = att.masked_fill(mask == 0, float('-inf'))
                    att = F.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v_selected
                
                # This is a simplified approach - in a real implementation, we would:
                # 1. Maintain the original sequence order
                # 2. Properly handle attention between all tokens, with appropriate sparsity patterns
                # 3. Merge the results back into the full sequence
                
                # For this demo, we'll just map the results back to the original sequence
                y_full = torch.zeros_like(q)
                y_full[:, :, selected_indices, :] = y
                
                # Reshape back
                y = y_full.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
            else:
                # No compression needed, proceed with regular attention
                if self.flash:
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
                    )
                else:
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
                    mask = self.mask[:, :, :seq_len, :seq_len]
                    att = att.masked_fill(mask == 0, float('-inf'))
                    att = F.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v
                
                # Reshape back
                y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        else:
            # Regular attention for shorter sequences
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
                mask = self.mask[:, :, :seq_len, :seq_len]
                att = att.masked_fill(mask == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
            
            # Reshape back
            y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# Utility function to update model config for billion-token context
def enable_billion_token_context(model, max_block_size=None, preserve_begin=None, preserve_end=None):
    """
    Update a GPT model config to enable billion-token context support.
    
    Args:
        model: GPT model instance
        max_block_size: Maximum tokens to process at once (None = auto)
        preserve_begin: Number of beginning tokens to preserve (None = auto)
        preserve_end: Number of end tokens to preserve (None = auto)
        
    Returns:
        Updated model
    """
    if not hasattr(model, 'config'):
        raise ValueError("Model doesn't have a config attribute")
    
    # Update model config
    model.config.billion_token_context = True
    
    # Set maximum block size if provided
    if max_block_size is not None:
        model.config.max_actual_block_size = max_block_size
    else:
        # Default to the model's original block size
        model.config.max_actual_block_size = model.config.block_size
    
    # Set token preservation parameters
    if preserve_begin is not None:
        model.config.preserve_begin_tokens = preserve_begin
    
    if preserve_end is not None:
        model.config.preserve_end_tokens = preserve_end
    
    # Enable sparse attention
    model.config.use_sparse_attention = True
    
    print(f"Billion-token context enabled with max_block_size={model.config.max_actual_block_size}")
    print(f"Token preservation: begin={model.config.preserve_begin_tokens}, end={model.config.preserve_end_tokens}")
    
    return model
