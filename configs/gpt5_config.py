"""
Configuration for gpt5 model (1000B parameters)
"""

class Gpt5Config(object):
    """Configuration for GPT-5 model"""
    
    def __init__(self):
        # Model architecture
        self.block_size = 1000000000
        self.vocab_size = 50304
        self.n_layer = 300
        self.n_head = 300
        self.n_embd = 22528
        self.dropout = 0.0
        self.bias = False  # False is slightly better
        
        # Parameter metrics
        self.parameter_count = "1000B"
    
        # Mixture of Experts
        self.use_moe = True
        self.num_experts = 300
        self.num_active_experts = 16
        self.expert_capacity_factor = 1.25
        self.moe_jitter_eps = 0.01
    
        # Unlimited context support
        self.billion_token_context = True
        self.use_sparse_attention = True
        self.preserve_begin_tokens = 0  # Auto-calculated
        self.preserve_end_tokens = 0  # Auto-calculated
        self.middle_token_compression_ratio = 0.1
        self.token_eviction_strategy = 'importance'