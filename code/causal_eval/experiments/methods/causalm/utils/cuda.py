import os
import torch.cuda as cuda

print('=' * 50)
print(f'cuda.is_available(): {cuda.is_available()}')
print(f'cuda.device_count(): {cuda.device_count()}')
print(f'os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}')
print()
print(f'>>> setting os.environ["CUDA_VISIBLE_DEVICES"] = "-1"')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
print()
print(f'cuda.is_available(): {cuda.is_available()}')
print(f'cuda.device_count(): {cuda.device_count()}')
print(f'os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}')
print('=' * 50)
