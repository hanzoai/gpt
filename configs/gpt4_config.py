"""
Configuration for gpt4 model (1.7T parameters)
"""

class Gpt4Config(object):
    """Configuration for GPT-4 model"""
    
    def __init__(self):
        # Model architecture
        self.block_size = 128000
        self.vocab_size = 50304
        self.n_layer = 200
        self.n_head = 200
        self.n_embd = 18432
        self.dropout = 0.0
        self.bias = False  # False is slightly better
        
        # Parameter metrics
        self.parameter_count = "1.7T"
        
        # Mixture of Experts
        self.use_moe = True
        self.num_experts = 200
        self.num_active_experts = 8
        self.expert_capacity_factor = 1.25
        self.moe_jitter_eps = 0.01