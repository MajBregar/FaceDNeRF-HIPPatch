import torch
from torch_utils.ops import upfirdn2d, bias_act, filtered_lrelu

# ---------------------------------------------------------------------
# Basic tensor setup.
x = torch.randn(1, 3, 64, 64, device='cuda')
f = torch.ones((4, 4), dtype=torch.float32, device='cuda') / 16

# ---------------------------------------------------------------------
print("Testing upfirdn2d...")
y = upfirdn2d.upfirdn2d(x, f)
print("upfirdn2d OK:", y.shape)
print("")

# ---------------------------------------------------------------------
print("Testing bias_act...")
b = torch.randn(3, device='cuda')
z = bias_act.bias_act(y, b)
print("bias_act OK:", z.shape)
print("")


# ---------------------------------------------------------------------

print("Testing filtered_lrelu (HIP backend)...")
x = torch.randn((1, 3, 64, 64), dtype=torch.float32, device='cuda')
fu = torch.ones((1, 1), dtype=torch.float32, device='cuda')
fd = torch.ones((1, 1), dtype=torch.float32, device='cuda')
b = torch.randn(3, dtype=torch.float32, device='cuda')

try:
    w = filtered_lrelu.filtered_lrelu(
        x,
        fu=fu,
        fd=fd,
        b=b,
        up=1,
        down=1,
        padding=0,
        gain=torch.sqrt(torch.tensor(2.0)),
        slope=0.2,
        clamp=None,
        flip_filter=False,
        impl='hip'
    )
    print("filtered_lrelu OK:", w.shape)
except Exception as e:
    print("filtered_lrelu FAILED:", str(e))


# print("Package structure:")
# print(dir(filtered_lrelu))
