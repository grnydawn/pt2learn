import torch
import torch.nn as nn
import triton.testing.do_bench as do_bench

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.fc1(x).relu() ** 2
        return self.fc2(x).relu() ** 2


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    mod = MLP().cuda()
    opt_mod = torch.compile(mod)
    x = torch.randn(1024, 1024, device="cuda")
    base_time, _, _ = do_bench(lambda: mod(x).sum().backward())
    opt_time, _, _ = do_bench(lambda: opt_mod(x).sum().backward())
    print(f"speedup: {base_time / opt_time:.2f}")
