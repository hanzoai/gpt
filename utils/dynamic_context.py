"""
Dynamic Context Manager for unlimited context length with intelligent token management.
This implementation allows for theoretically unlimited context with middle token eviction
when memory constraints are reached.
"""

import torch
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

@dataclass
class DynamicContextConfig:
    """Configuration for the dynamic context manager."""
    # Memory management
    max_memory_mb: int = 0  # 0 = auto-detect (80% of available system memory)
    check_memory_every_n_tokens: int = 100  # How often to check memory usage
    
    # Token preservation
    begin_tokens_to_preserve: int = 0  # 0 = auto (10% of context)
    end_tokens_to_preserve: int = 0  # 0 = auto (20% of context)
    min_middle_tokens_to_keep: int = 0  # 0 = auto (max(100, 10% of context))
    
    # Importance scoring
    recency_weight: float = 1.0  # Weight for token recency in importance scoring
    attention_weight: float = 2.0  # Weight for attention patterns
    content_weight: float = 1.5  # Weight for content-based importance
    min_importance_threshold: float = 0.2  # Minimum threshold to keep a token
    
    # Initial buffer sizes
    initial_buffer_size: int = 4096  # Initial size of token buffer
    buffer_growth_factor: float = 2.0  # How much to grow buffer when needed
    
    # Experimental
    use_token_compression: bool = False  # Use compression instead of eviction
    semantic_landmark_detection: bool = False  # Detect and preserve semantic landmarks
    
    def __post_init__(self):
        """Auto-detect settings if needed."""
        if self.max_memory_mb == 0:
            try:
                import psutil
                mem = psutil.virtual_memory()
                self.max_memory_mb = int(mem.available * 0.8 / (1024 * 1024))
                print(f"Auto-configured max_memory_mb to {self.max_memory_mb}")
            except ImportError:
                self.max_memory_mb = 4096  # Default to 4GB if psutil not available
                print(f"psutil not available, defaulting max_memory_mb to {self.max_memory_mb}")


class CircularTokenBuffer:
    """
    Circular buffer for token storage with efficient access to beginning, middle and end.
    """
    
    def __init__(self, initial_size: int):
        """Initialize the buffer with given size."""
        self.buffer = [None] * initial_size
        self.head = 0
        self.tail = 0
        self.size = 0
        self.capacity = initial_size
        self.token_metadata = {}  # Additional token information
    
    def append(self, tokens: List[Any]) -> None:
        """Add tokens to the buffer, resizing if needed."""
        if not tokens:
            return
            
        # Resize if needed
        if self.size + len(tokens) > self.capacity:
            new_capacity = max(self.capacity * 2, self.size + len(tokens))
            self._resize(new_capacity)
        
        # Add tokens
        for token in tokens:
            self.buffer[self.tail] = token
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
    
    def get_first(self, n: int) -> List[Any]:
        """Get first n tokens from buffer."""
        result = []
        idx = self.head
        count = min(n, self.size)
        
        for _ in range(count):
            result.append(self.buffer[idx])
            idx = (idx + 1) % self.capacity
            
        return result
    
    def get_last(self, n: int) -> List[Any]:
        """Get last n tokens from buffer."""
        result = []
        count = min(n, self.size)
        
        if count == 0:
            return result
            
        idx = (self.tail - 1) % self.capacity  # Start from the last element
        for _ in range(count):
            result.append(self.buffer[idx])
            idx = (idx - 1) % self.capacity
            
        return result[::-1]  # Reverse to maintain order
    
    def get_middle(self, begin_count: int, end_count: int) -> List[Any]:
        """Get all tokens between first begin_count and last end_count tokens."""
        begin = min(begin_count, self.size)
        end = min(end_count, self.size - begin)
        middle_count = self.size - begin - end
        
        if middle_count <= 0:
            return []
            
        result = []
        idx = (self.head + begin) % self.capacity
        for _ in range(middle_count):
            result.append(self.buffer[idx])
            idx = (idx + 1) % self.capacity
            
        return result
    
    def get_all(self) -> List[Any]:
        """Get all tokens in order."""
        if self.size == 0:
            return []
            
        result = []
        idx = self.head
        for _ in range(self.size):
            result.append(self.buffer[idx])
            idx = (idx + 1) % self.capacity
            
        return result
    
    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer = [None] * self.capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.token_metadata = {}
    
    def _resize(self, new_capacity: int) -> None:
        """Resize the buffer to new capacity."""
        new_buffer = [None] * new_capacity
        
        # Copy existing elements
        idx = self.head
        for i in range(self.size):
            new_buffer[i] = self.buffer[idx]
            idx = (idx + 1) % self.capacity
        
        self.buffer = new_buffer
        self.head = 0
        self.tail = self.size
        self.capacity = new_capacity
        
    def reconstruct(self, begin_tokens: List[Any], middle_tokens: List[Any], end_tokens: List[Any]) -> None:
        """Reconstruct buffer with specified tokens."""
        total_size = len(begin_tokens) + len(middle_tokens) + len(end_tokens)
        
        # Ensure capacity is sufficient
        if total_size > self.capacity:
            self._resize(max(self.capacity * 2, total_size))
        
        # Reset and refill
        self.reset()
        
        # Add all tokens
        all_tokens = begin_tokens + middle_tokens + end_tokens
        self.append(all_tokens)


class TokenImportanceTracker:
    """
    Tracks token importance based on multiple factors: recency, attention, and content.
    """
    
    def __init__(self, config: DynamicContextConfig):
        """Initialize the importance tracker."""
        self.config = config
        self.token_scores = {}  # Map token IDs to importance scores
        self.token_metadata = {}  # Additional token information
        self.attention_history = []  # Track recent attention patterns
        self.position_map = {}  # Map token to its position
        self.age_map = {}  # Map token to its age (in steps)
        self.current_step = 0
    
    def update(self, tokens: List[Any], positions: List[int], attention_weights: Optional[torch.Tensor] = None) -> None:
        """
        Update importance scores based on new information.
        
        Args:
            tokens: List of token IDs
            positions: Positions of tokens in the sequence
            attention_weights: Optional tensor of attention weights
        """
        self.current_step += 1
        
        # Update position and age information
        for token, pos in zip(tokens, positions):
            self.position_map[token] = pos
            self.age_map[token] = self.current_step
        
        # Update importance scores based on multiple factors
        for i, token in enumerate(tokens):
            # 1. Recency score (newer tokens tend to be more important)
            recency_score = self.config.recency_weight * (1.0 - i / max(1, len(tokens)))
            
            # 2. Attention score (tokens that receive more attention are important)
            attention_score = 0.0
            if attention_weights is not None:
                attention_score = self._calculate_attention_score(token, attention_weights)
            attention_score *= self.config.attention_weight
            
            # 3. Content score (some tokens are inherently more important)
            content_score = self._calculate_content_score(token) * self.config.content_weight
            
            # Combine scores
            total_score = recency_score + attention_score + content_score
            self.token_scores[token] = total_score
    
    def _calculate_attention_score(self, token, attention_weights):
        """Calculate importance based on attention patterns."""
        # This would use actual attention weights from the model
        # For now, use a simple placeholder that favors tokens with mid-range positions
        token_pos = self.position_map.get(token, 0)
        # Simplified attention score - middle positions get higher scores
        # In a real implementation, this would use actual attention statistics
        return 0.5  # Placeholder
    
    def _calculate_content_score(self, token):
        """
        Calculate importance based on token content.
        Some tokens are inherently more important (questions, entities, etc.)
        """
        # This would analyze the token itself for importance
        # Placeholder implementation - would be more sophisticated in practice
        # Special tokens, entities, rare tokens might get higher scores
        return 0.3  # Placeholder
    
    def rank_tokens(self, tokens: List[Any]) -> List[Tuple[Any, float]]:
        """Rank tokens by importance, returning (token, score) pairs."""
        scored_tokens = [(token, self.token_scores.get(token, 0.0)) for token in tokens]
        return sorted(scored_tokens, key=lambda x: x[1], reverse=True)
    
    def filter_by_importance(self, tokens: List[Any], threshold: float) -> List[Any]:
        """Filter tokens by importance threshold."""
        return [t for t in tokens if self.token_scores.get(t, 0.0) >= threshold]
    
    def get_top_k(self, tokens: List[Any], k: int) -> List[Any]:
        """Get top k tokens by importance."""
        ranked = self.rank_tokens(tokens)
        return [t for t, _ in ranked[:k]]


class MemoryMonitor:
    """
    Monitors memory usage and signals when compression/eviction is needed.
    """
    
    def __init__(self, config: DynamicContextConfig):
        """Initialize the memory monitor."""
        self.config = config
        self.check_frequency = config.check_memory_every_n_tokens
        self.last_check = 0
        self.memory_limit_mb = config.max_memory_mb
        self.memory_usage_history = []
    
    def is_memory_constrained(self) -> bool:
        """Check if memory usage is approaching limits."""
        self.last_check += 1
        
        # Only check periodically to reduce overhead
        if self.last_check % self.check_frequency != 0:
            return False
        
        current_usage_mb = self._get_current_memory_usage_mb()
        self.memory_usage_history.append((time.time(), current_usage_mb))
        
        # If using more than 90% of allocated memory, signal constraint
        return current_usage_mb > self.memory_limit_mb * 0.9
    
    def _get_current_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / (1024 * 1024)  # Convert bytes to MB
        except ImportError:
            # If psutil not available, return a percentage of max memory
            # This is not accurate but allows the code to run
            return self.memory_limit_mb * 0.5  # Assume 50% usage
    
    def get_memory_pressure(self) -> float:
        """Get memory pressure level between 0.0 and 1.0."""
        current_usage = self._get_current_memory_usage_mb()
        return min(1.0, current_usage / max(1.0, self.memory_limit_mb))
    
    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        current_usage = self._get_current_memory_usage_mb()
        return max(0.0, self.memory_limit_mb - current_usage)


class DynamicContextManager:
    """
    Manages context with unlimited length, dropping middle tokens when memory is constrained.
    """
    
    def __init__(self, config: DynamicContextConfig):
        """Initialize the context manager."""
        self.config = config
        
        # Set up token storage
        self.token_buffer = CircularTokenBuffer(config.initial_buffer_size)
        self.importance_tracker = TokenImportanceTracker(config)
        self.memory_monitor = MemoryMonitor(config)
        
        # Statistics and monitoring
        self.total_tokens_seen = 0
        self.total_tokens_evicted = 0
        self.eviction_events = []
    
    def add_tokens(self, new_tokens: List[Any], positions: Optional[List[int]] = None, 
                  attention_weights: Optional[torch.Tensor] = None) -> None:
        """
        Add new tokens to the context, evicting middle tokens if needed.
        
        Args:
            new_tokens: Tokens to add
            positions: Optional positions of tokens
            attention_weights: Optional attention weights
        """
        if not new_tokens:
            return
        
        # Generate positions if not provided
        if positions is None:
            start_pos = self.total_tokens_seen
            positions = list(range(start_pos, start_pos + len(new_tokens)))
        
        # Update token importance
        self.importance_tracker.update(new_tokens, positions, attention_weights)
        
        # Check if memory is constrained and evict if needed
        if self.memory_monitor.is_memory_constrained():
            self._evict_middle_tokens()
        
        # Add new tokens to buffer
        self.token_buffer.append(new_tokens)
        self.total_tokens_seen += len(new_tokens)
    
    def _evict_middle_tokens(self) -> int:
        """
        Evict middle tokens based on importance, preserving beginning and end.
        Returns number of tokens evicted.
        """
        # Determine how many tokens to preserve at beginning and end
        begin_count = self.config.begin_tokens_to_preserve
        if begin_count == 0:
            begin_count = max(100, int(self.token_buffer.size * 0.1))
            
        end_count = self.config.end_tokens_to_preserve
        if end_count == 0:
            end_count = max(100, int(self.token_buffer.size * 0.2))
        
        # Make sure we're not trying to preserve more than we have
        total_to_preserve = begin_count + end_count
        if total_to_preserve >= self.token_buffer.size:
            scaling_factor = self.token_buffer.size / total_to_preserve * 0.9
            begin_count = int(begin_count * scaling_factor)
            end_count = int(end_count * scaling_factor)
        
        # Get beginning, middle, and end sections
        begin_tokens = self.token_buffer.get_first(begin_count)
        end_tokens = self.token_buffer.get_last(end_count)
        middle_tokens = self.token_buffer.get_middle(begin_count, end_count)
        
        # No middle tokens to evict
        if not middle_tokens:
            return 0
        
        # Determine how many middle tokens to keep based on importance and memory
        memory_pressure = self.memory_monitor.get_memory_pressure()
        middle_retention_ratio = max(0.1, 1.0 - memory_pressure)  # Keep at least 10%
        
        # Sort middle by importance
        ranked_tokens = self.importance_tracker.rank_tokens(middle_tokens)
        
        # Calculate how many to keep
        tokens_to_keep_count = max(
            self.config.min_middle_tokens_to_keep,
            int(len(middle_tokens) * middle_retention_ratio)
        )
        
        # Select top tokens to keep
        tokens_to_keep = [token for token, score in ranked_tokens[:tokens_to_keep_count]]
        
        # Calculate how many were evicted
        evicted_count = len(middle_tokens) - len(tokens_to_keep)
        
        # Record eviction event
        if evicted_count > 0:
            self.eviction_events.append({
                'timestamp': time.time(),
                'context_size': self.token_buffer.size,
                'evicted_count': evicted_count,
                'memory_pressure': memory_pressure,
            })
            self.total_tokens_evicted += evicted_count
            
            print(f"Evicted {evicted_count} middle tokens, keeping {begin_count} beginning, "
                  f"{len(tokens_to_keep)} middle, and {end_count} end tokens.")
        
        # Reconstruct buffer with kept tokens
        self.token_buffer.reconstruct(begin_tokens, tokens_to_keep, end_tokens)
        
        return evicted_count
    
    def get_all_tokens(self) -> List[Any]:
        """Get all tokens in the context."""
        return self.token_buffer.get_all()
    
    def get_context_size(self) -> int:
        """Get current context size."""
        return self.token_buffer.size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the context manager."""
        return {
            'total_tokens_seen': self.total_tokens_seen,
            'total_tokens_evicted': self.total_tokens_evicted,
            'current_context_size': self.token_buffer.size,
            'eviction_events': len(self.eviction_events),
            'memory_pressure': self.memory_monitor.get_memory_pressure(),
            'available_memory_mb': self.memory_monitor.get_available_memory_mb(),
        }


def create_context_manager(max_memory_mb=0, begin_ratio=0.1, end_ratio=0.2):
    """
    Create a dynamic context manager with simple configuration.
    
    Args:
        max_memory_mb: Maximum memory to use (0 = auto-detect)
        begin_ratio: Ratio of context to preserve at beginning
        end_ratio: Ratio of context to preserve at end
        
    Returns:
        Configured DynamicContextManager
    """
    config = DynamicContextConfig(max_memory_mb=max_memory_mb)
    
    # Set preservation ratios instead of absolute counts
    # Actual token counts will be calculated dynamically
    config.begin_tokens_to_preserve = 0  # Auto calculated as begin_ratio * context_size
    config.end_tokens_to_preserve = 0  # Auto calculated as end_ratio * context_size
    
    # Create manager with custom importance weights
    manager = DynamicContextManager(config)
    
    return manager
