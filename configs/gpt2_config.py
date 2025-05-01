"""
Configuration for gpt2 model (124M parameters)
"""

class Gpt2Config(object):
    """Configuration for GPT-2 model"""
    
    def __init__(self):
        # Model architecture
        self.block_size = 1024
        self.vocab_size = 50304
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.0
        self.bias = False  # False is slightly better
        
        # Parameter metrics
        self.parameter_count = "124M"