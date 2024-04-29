import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(3, 3).to(device)

# Check the device of the tensor
if tensor.device.type == 'cuda':
    print("Tensor was created on the GPU")
else:
    print("Tensor was created on the CPU")
