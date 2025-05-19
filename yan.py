import torch
import torch.nn as nn
import torch.nn.functional as F


# Comment: 
# 1. It makes no sense to include 'input_size' and 'output_size' as parameters only. 
#    The beauty of this arch is in its ability to decide "pathways" across neurons of "any # of layers"
#    True operation should be more similar to specifying "# of hidden neurons", and learn weights (input_size + hidden_size + output_size, input_size + hidden_size + output_size)
#    where input->xxx should be positive or zero, and xxx->output should be positive or zero. 
#    the intricate connection between hidden neurons are the joy of this architecture. 
#    - we could limit the number of propagation in practice, this effectively bypass the issue of cyclic connection. 
#    - we basically collect activation in output neurons as "output". 


import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_steps=5):
        super(PathwayNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_size = input_size + hidden_size + output_size
        self.max_steps = max_steps  # Maximum propagation steps
        
        # Initialize the full weight matrix for all neurons
        self.weight_params = nn.Parameter(torch.randn(self.total_size, self.total_size))
        self.bias = nn.Parameter(torch.zeros(self.total_size))
        
        # Create mask to enforce antisymmetric weights
        mask = torch.ones(self.total_size, self.total_size)
        mask = torch.triu(mask, 1) - torch.tril(mask, -1)  # Upper triangular - lower triangular
        self.register_buffer('antisym_mask', mask)
        
        # input & output connectivity mask
        connectivity_mask = torch.ones(self.total_size, self.total_size)
        connectivity_mask[:, :input_size] = 0
        output_start = input_size + hidden_size
        connectivity_mask[output_start:, :] = 0
        self.register_buffer('connectivity_mask', connectivity_mask)

    def get_weights(self):
        # Apply mask to enforce antisymmetry: w(i→j) = -w(j→i)
        raw_weights = self.weight_params * self.antisym_mask
        weights = raw_weights - raw_weights.transpose(0, 1)
        
        # Apply connectivity constraints
        weights = weights * self.connectivity_mask
        
        return weights
        
    def forward(self, x):
        batch_size = x.size(0)
        weights = self.get_weights()
        
        full_activation = torch.zeros(batch_size, self.total_size, device=x.device)
        full_activation[:, :self.input_size] = x  # Set input activations
        
        # Get positive weights only (viable pathways)
        positive_weights = torch.clamp(weights, min=0)  # Ensure weights are positive
        
        for _ in range(self.max_steps):
            # resistance propagation: x - x*W
            activation_change = torch.matmul(full_activation, positive_weights)
            full_activation = full_activation - activation_change + self.bias
            full_activation = F.relu(full_activation)

        output = full_activation[:, -self.output_size:]
        return output
    
    def visualize_connectivity(self):
        """Visualize the learned connectivity pattern"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        weights = self.get_weights()
        positive_weights = (weights > 0).float() * weights
        
        # Plot the connectivity pattern
        plt.figure(figsize=(10, 10))
        plt.imshow(positive_weights.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        
        # Add boundary lines between input, hidden, and output regions
        plt.axhline(y=self.input_size-0.5, color='blue', linestyle='-', alpha=0.3)
        plt.axvline(x=self.input_size-0.5, color='blue', linestyle='-', alpha=0.3)
        
        output_start = self.input_size + self.hidden_size
        plt.axhline(y=output_start-0.5, color='blue', linestyle='-', alpha=0.3)
        plt.axvline(x=output_start-0.5, color='blue', linestyle='-', alpha=0.3)
        
        # Label regions
        plt.text(self.input_size//2, -2, "Input", ha='center')
        plt.text(self.input_size + self.hidden_size//2, -2, "Hidden", ha='center')
        plt.text(self.input_size + self.hidden_size + self.output_size//2, -2, "Output", ha='center')
        
        plt.title("Learned Connectivity Pattern (Positive Weights Only)")
        plt.tight_layout()
        plt.show()
        
        # Also visualize the activation flow over steps for a sample input
        if hasattr(self, 'last_activations'):
            plt.figure(figsize=(12, 6))
            for step, act in enumerate(self.last_activations):
                plt.subplot(1, len(self.last_activations), step+1)
                plt.imshow(act[0].detach().cpu().numpy().reshape(-1, 1), 
                          cmap='viridis', aspect='auto')
                plt.title(f"Step {step}")
            plt.tight_layout()
            plt.show()
        
        return positive_weights

    def trace_activation_flow(self, x):
        """Trace and visualize activation flow for a specific input"""

        batch_size = x.size(0)
        weights = self.get_weights()
        
        full_activation = torch.zeros(batch_size, self.total_size, device=x.device)
        full_activation[:, :self.input_size] = x  # Set input activations
        
        # Get positive weights only (viable pathways)
        positive_weights = torch.clamp(weights, min=0)  # Ensure weights are positive

        # Store activations at each step
        self.last_activations = [full_activation.clone()]
        
        # Propagate activations
        for _ in range(self.max_steps):
            # resistance propagation: x - x*W
            activation_change = torch.matmul(full_activation, positive_weights)
            full_activation = full_activation - activation_change + self.bias
            full_activation = F.relu(full_activation)
            
            self.last_activations.append(full_activation.clone())
        
        # Visualize activation flow
        self.visualize_connectivity()
        
        return full_activation[:, -self.output_size:]
