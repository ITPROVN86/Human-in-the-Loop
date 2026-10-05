import sys
try:
    import torch
except Exception as exc:
    print(f"ERROR: cannot import torch: {exc}")
    raise SystemExit(1)

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}; VRAM={p.total_memory/1024**3:.1f} GB; capability={p.major}.{p.minor}")
    x=torch.randn(2048,2048,device="cuda")
    y=x@x
    torch.cuda.synchronize()
    print("CUDA matrix test: PASS", tuple(y.shape))
else:
    print("CUDA matrix test: FAIL — install a CUDA-enabled PyTorch build and check the NVIDIA driver.")
    raise SystemExit(2)

