# Investigating the Parameter
# We can loop through the state_dict to see all parameters and their shapes
import torch

# Load the checkpoint
checkpoint = torch.load('/opt/repository/gpt2_safetensors/pytorch_model.bin',
                        map_location=torch.device('cpu'))

# Extract the state_dict


# Loop through and print details about each parameter
for key, value in checkpoint.items():
    print(f'{key}: {value.shape if hasattr(value, "shape") else len(value)}')
