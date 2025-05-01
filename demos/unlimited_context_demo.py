"""
Demo of unlimited context capability with middle token eviction.
This script demonstrates how the dynamic context manager works with token eviction
while preserving beginning and end tokens.
"""

import torch
import time
import random
import sys
from dynamic_context import DynamicContextConfig, DynamicContextManager

# Create a simple token generator for testing
def generate_random_tokens(count, vocab_size=50000):
    """Generate random tokens for testing."""
    return [random.randint(0, vocab_size-1) for _ in range(count)]

def print_colored(text, color=None):
    """Print colored text if supported."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    
    if color in colors and sys.stdout.isatty():
        print(colors[color] + text + colors['end'])
    else:
        print(text)

def run_demo(max_memory_mb=100, total_tokens=100000, batch_size=1000, forced_eviction=True):
    """
    Run a demonstration of the dynamic context manager.
    
    Args:
        max_memory_mb: Maximum memory to use (artificially low for demo)
        total_tokens: Total number of tokens to generate
        batch_size: How many tokens to add at once
        forced_eviction: Force eviction on every batch for demonstration
    """
    # Create configuration with artificially low memory limit
    config = DynamicContextConfig(
        max_memory_mb=max_memory_mb,  # Artificially low for demo
        begin_tokens_to_preserve=int(total_tokens * 0.1),  # 10% beginning
        end_tokens_to_preserve=int(total_tokens * 0.2),    # 20% end
        check_memory_every_n_tokens=batch_size  # Check after each batch
    )
    
    # Create context manager
    context_manager = DynamicContextManager(config)
    
    # Generate and add tokens in batches
    print_colored("Starting demo with max_memory={}MB, total_tokens={}".format(max_memory_mb, total_tokens), "blue")
    print_colored("Will preserve {} beginning and {} end tokens".format(config.begin_tokens_to_preserve, config.end_tokens_to_preserve), "blue")
    
    begin_tokens = []
    middle_tokens = []
    end_tokens = []
    
    start_time = time.time()
    
    # Generate tokens in 3 phases (beginning, middle, end)
    begin_count = int(total_tokens * 0.1)
    end_count = int(total_tokens * 0.2)
    middle_count = total_tokens - begin_count - end_count
    
    # Generate beginning tokens (mark as special for identification)
    begin_tokens = [-(i+1) for i in range(begin_count)]  # Negative tokens for beginning
    
    # Generate middle tokens
    middle_tokens = list(range(middle_count))  # Sequential tokens for middle
    
    # Generate end tokens (mark as special for identification)
    end_tokens = [1000000 + i for i in range(end_count)]  # Large tokens for end
    
    # Combined token stream
    all_tokens = begin_tokens + middle_tokens + end_tokens
    
    # Process in batches
    for i in range(0, len(all_tokens), batch_size):
        batch = all_tokens[i:i+batch_size]
        
        # If forcing eviction, simulate memory pressure
        if forced_eviction:
            context_manager.memory_monitor.is_memory_constrained = lambda: True
        
        # Add batch
        context_manager.add_tokens(batch)
        
        # Log progress
        context_size = context_manager.get_context_size()
        print("Added batch {}/{}, context size: {}".format(i//batch_size + 1, (len(all_tokens) + batch_size - 1)//batch_size, context_size))
        
        # Every few batches, show a sample of tokens
        if (i//batch_size) % 5 == 0:
            current_tokens = context_manager.get_all_tokens()
            
            # Print summary
            print_colored("\nCurrent context ({} tokens):".format(len(current_tokens)), "green")
            
            # Count token types to verify preservation
            begin_preserved = sum(1 for t in current_tokens if t < 0)
            end_preserved = sum(1 for t in current_tokens if t >= 1000000)
            middle_preserved = len(current_tokens) - begin_preserved - end_preserved
            
            print_colored("Beginning tokens: {}/{} ({:.1f}%)".format(begin_preserved, begin_count, begin_preserved/begin_count*100), 
                       "green" if begin_preserved == begin_count else "yellow")
            print_colored("Middle tokens: {}/{} ({:.1f}%)".format(middle_preserved, middle_count, middle_preserved/middle_count*100), "yellow")
            print_colored("End tokens: {}/{} ({:.1f}%)".format(end_preserved, end_count, end_preserved/end_count*100), 
                       "green" if end_preserved == end_count else "yellow")
            
            if len(current_tokens) > 0:
                # Sample beginning
                print("\nSample of beginning tokens:", end=" ")
                for t in current_tokens[:10]:
                    if t < 0:
                        print("\033[94m{}\033[0m".format(t), end=" ")  # Blue for beginning
                    elif t >= 1000000:
                        print("\033[92m{}\033[0m".format(t), end=" ")  # Green for end
                    else:
                        print(t, end=" ")  # Default for middle
                
                # Sample middle (if any remain)
                if middle_preserved > 0:
                    middle_indices = [i for i, t in enumerate(current_tokens) if 0 <= t < 1000000]
                    if middle_indices:
                        mid_point = middle_indices[len(middle_indices)//2]
                        print("\nSample of middle tokens:", end=" ")
                        sample_range = range(max(0, mid_point-5), min(len(current_tokens), mid_point+5))
                        for i in sample_range:
                            t = current_tokens[i]
                            if t < 0:
                                print("\033[94m{}\033[0m".format(t), end=" ")  # Blue for beginning
                            elif t >= 1000000:
                                print("\033[92m{}\033[0m".format(t), end=" ")  # Green for end
                            else:
                                print(t, end=" ")  # Default for middle
                
                # Sample end
                print("\nSample of end tokens:", end=" ")
                for t in current_tokens[-10:]:
                    if t < 0:
                        print("\033[94m{}\033[0m".format(t), end=" ")  # Blue for beginning
                    elif t >= 1000000:
                        print("\033[92m{}\033[0m".format(t), end=" ")  # Green for end
                    else:
                        print(t, end=" ")  # Default for middle
                print("\n")
    
    # Final stats
    end_time = time.time()
    elapsed = end_time - start_time
    
    print_colored("\nDemo completed in {:.2f} seconds".format(elapsed), "magenta")
    print_colored("Total tokens processed: {}".format(total_tokens), "magenta")
    print_colored("Final context size: {} tokens".format(context_manager.get_context_size()), "magenta")
    print_colored("Total tokens evicted: {}".format(context_manager.total_tokens_evicted), "magenta")
    print_colored("Eviction events: {}".format(len(context_manager.eviction_events)), "magenta")
    
    # Verify beginning and end preservation
    current_tokens = context_manager.get_all_tokens()
    begin_preserved = sum(1 for t in current_tokens if t < 0)
    end_preserved = sum(1 for t in current_tokens if t >= 1000000)
    
    success = (begin_preserved == begin_count and end_preserved == end_count)
    
    if success:
        print_colored("\n✅ SUCCESS: All beginning and end tokens were preserved", "green")
    else:
        print_colored("\n❌ FAILURE: Some beginning or end tokens were lost", "red")
        print_colored("Beginning tokens: {}/{}".format(begin_preserved, begin_count), "red" if begin_preserved < begin_count else "green")
        print_colored("End tokens: {}/{}".format(end_preserved, end_count), "red" if end_preserved < end_count else "green")
    
    return success

if __name__ == "__main__":
    # Run the demo
    print_colored("=" * 80, "cyan")
    print_colored("UNLIMITED CONTEXT DEMO WITH MIDDLE TOKEN EVICTION", "cyan")
    print_colored("=" * 80, "cyan")
    
    # Run with smaller numbers for quick testing
    quick_test_success = run_demo(max_memory_mb=10, total_tokens=1000, batch_size=100)
    
    if quick_test_success:
        print_colored("\nQuick test passed successfully! Would you like to run with larger token counts? (y/n)", "blue")
        response = input().strip().lower()
        
        if response == 'y':
            print_colored("\nRunning full demo with 1,000,000 tokens...", "blue")
            run_demo(max_memory_mb=100, total_tokens=1000000, batch_size=10000)
