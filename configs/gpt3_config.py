"""
Configuration for gpt3 model (175B parameters)
"""

class Gpt3Config(object):
    """Configuration for GPT-3 model"""
    
    def __init__(self):
        # Model architecture
        self.block_size = 2048
        self.vocab_size = 50304
        self.n_layer = 96
        self.n_head = 96
        self.n_embd = 12288
        self.dropout = 0.0
        self.bias = False  # False is slightly better
        
        # Parameter metrics
        self.parameter_count = "175B"