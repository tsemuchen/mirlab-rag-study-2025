import torch

device_id = 0  # or 1, 2, ...
device_str = f"cuda:{device_id}"

try:
    x = torch.tensor([1, 2, 3]).to(device_str)
    print(f"Success on device {device_str}: {x}")
except Exception as e:
    print(f"Failed to allocate on {device_str}: {e}")
