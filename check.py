import torch
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.get_device_name(0))  # Should display your GPU name (e.g., RTX 4050)