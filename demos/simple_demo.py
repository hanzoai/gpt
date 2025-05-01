"""
Simple demonstration of unlimited token context with middle eviction.
This script shows how we can retain beginning and end tokens while
evicting middle tokens when context gets too large.
"""

import random
import time

class SimpleContextManager:
    """
    A simplified context manager that demonstrates unlimited context
    with middle token eviction.
    """
    
    def __init__(self, max_tokens=1000, preserve_begin=100, preserve_end=200):
        """Initialize with max tokens to keep in memory."""
        self.max_tokens = max_tokens
        self.preserve_begin = preserve_begin
        self.preserve_end = preserve_end
        self.tokens = []
        self.evicted_count = 0
    
    def add_tokens(self, new_tokens):
        """Add tokens, evicting middle tokens if necessary."""
        self.tokens.extend(new_tokens)
        
        # If we exceed max tokens, evict from the middle
        if len(self.tokens) > self.max_tokens:
            self._evict_middle_tokens()
    
    def _evict_middle_tokens(self):
        """Evict tokens from the middle of the context."""
        # Calculate how many tokens to keep
        total_preserve = self.preserve_begin + self.preserve_end
        if total_preserve >= len(self.tokens):
            # Can't preserve more than we have
            ratio = float(len(self.tokens) - 10) / total_preserve
            self.preserve_begin = max(5, int(self.preserve_begin * ratio))
            self.preserve_end = max(5, int(self.preserve_end * ratio))
            total_preserve = self.preserve_begin + self.preserve_end
        
        # Calculate how many to evict
        to_keep = min(self.max_tokens, len(self.tokens))
        middle_tokens_to_keep = to_keep - total_preserve
        
        # Get beginning, middle and end sections
        beginning = self.tokens[:self.preserve_begin]
        ending = self.tokens[-self.preserve_end:]
        
        # Middle section excludes beginning and end
        full_middle = self.tokens[self.preserve_begin:-self.preserve_end] if self.preserve_end > 0 else self.tokens[self.preserve_begin:]
        
        # Choose which middle tokens to keep (random selection for simplicity)
        # In a real implementation, we'd use importance scores
        if len(full_middle) > middle_tokens_to_keep and middle_tokens_to_keep > 0:
            # Random sample of middle tokens to keep
            step = float(len(full_middle)) / middle_tokens_to_keep
            indices_to_keep = [int(i * step) for i in range(middle_tokens_to_keep)]
            middle_to_keep = [full_middle[i] for i in indices_to_keep]
            
            # Track how many we evicted
            evicted = len(full_middle) - len(middle_to_keep)
            self.evicted_count += evicted
            
            # Rebuild the token list
            self.tokens = beginning + middle_to_keep + ending
            
            print("Evicted", evicted, "tokens from the middle")
            print("Now have", len(self.tokens), "tokens in context")
    
    def get_stats(self):
        """Get statistics about token distribution."""
        if not self.tokens:
            return "No tokens in context"
        
        # Count beginning, middle and end tokens
        beginning_count = min(self.preserve_begin, len(self.tokens))
        ending_count = min(self.preserve_end, len(self.tokens) - beginning_count)
        middle_count = len(self.tokens) - beginning_count - ending_count
        
        stats = "Total tokens: {}\n".format(len(self.tokens))
        stats += "Beginning tokens: {}\n".format(beginning_count)
        stats += "Middle tokens: {}\n".format(middle_count)
        stats += "Ending tokens: {}\n".format(ending_count)
        stats += "Total evicted: {}\n".format(self.evicted_count)
        
        return stats

def main():
    """Run a simple demonstration."""
    print("=" * 60)
    print("UNLIMITED CONTEXT WITH MIDDLE TOKEN EVICTION")
    print("=" * 60)
    
    # Create a context manager with small limits for demo purposes
    context = SimpleContextManager(max_tokens=100, preserve_begin=20, preserve_end=40)
    
    # Create distinctive token types
    begin_tokens = [-i for i in range(1, 31)]  # Negative tokens for beginning
    middle_tokens = list(range(1, 201))  # Regular numbers for middle
    end_tokens = [1000 + i for i in range(1, 51)]  # Large numbers for end
    
    # Add tokens in phases
    print("Adding beginning tokens...")
    context.add_tokens(begin_tokens[:20])
    print(context.get_stats())
    
    print("\nAdding first batch of middle tokens...")
    context.add_tokens(middle_tokens[:50])
    print(context.get_stats())
    
    print("\nAdding more middle tokens (should trigger eviction)...")
    context.add_tokens(middle_tokens[50:150])
    print(context.get_stats())
    
    print("\nAdding end tokens...")
    context.add_tokens(end_tokens)
    print(context.get_stats())
    
    # Display some sample tokens to verify preservation
    if context.tokens:
        print("\nSample of beginning tokens:", context.tokens[:5])
        
        # Find some middle tokens to display
        middle_start = min(context.preserve_begin, len(context.tokens))
        middle_end = max(0, len(context.tokens) - context.preserve_end)
        middle_sample = context.tokens[middle_start:middle_end]
        if middle_sample:
            print("Sample of middle tokens:", middle_sample[:5], "...")
        
        end_start = max(0, len(context.tokens) - min(context.preserve_end, len(context.tokens)))
        print("Sample of ending tokens:", context.tokens[-5:])
    
    # Verify beginning and end preservation
    beginning_preserved = sum(1 for t in context.tokens[:20] if t < 0)
    end_preserved = sum(1 for t in context.tokens[-40:] if t >= 1000)
    
    print("\nVerification:")
    print("Beginning tokens preserved: {}/{}".format(beginning_preserved, min(20, len(begin_tokens))))
    print("End tokens preserved: {}/{}".format(end_preserved, min(40, len(end_tokens))))
    
    if beginning_preserved == min(20, len(begin_tokens)) and end_preserved == min(40, len(end_tokens)):
        print("\nSUCCESS: All beginning and end tokens were preserved!")
    else:
        print("\nFAILURE: Some beginning or end tokens were lost.")

if __name__ == "__main__":
    main()
