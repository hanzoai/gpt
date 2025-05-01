"""
Configuration for gpt4_5 model (3T parameters)
"""

class Gpt4_5Config(object):
    """Configuration for GPT-4.5 model"""
    
    def __init__(self):
        # Model architecture
        self.block_size = 512000
        self.vocab_size = 50304
        self.n_layer = 240
        self.n_head = 240
        self.n_embd = 20480
        self.dropout = 0.0
        self.bias = False  # False is slightly better
        
        # Parameter metrics
        self.parameter_count = "3T"
        
        # Mixture of Experts
        self.use_moe = True
        self.num_experts = 240
        self.num_active_experts = 12
        self.expert_capacity_factor = 1.25
        self.moe_jitter_eps = 0.01