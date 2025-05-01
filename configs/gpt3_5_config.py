"""
Configuration for gpt3_5 model (350B parameters)
"""

class Gpt3_5Config(object):
    """Configuration for GPT-3.5 model"""
    
    def __init__(self):
        # Model architecture
        self.block_size = 16384
        self.vocab_size = 50304
        self.n_layer = 120
        self.n_head = 120
        self.n_embd = 14336
        self.dropout = 0.0
        self.bias = False  # False is slightly better
        
        # Parameter metrics
        self.parameter_count = "350B"