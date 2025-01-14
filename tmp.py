import torch
import torch.nn.functional as F

r1 = 0.3
r2 = 0.5
rs = 0.7
rss = 0.7
ref = 0

src = torch.Tensor([r1, r2, rs, rss, ref])
output = F.softmax(src, dim=0)
print(output)
src = torch.Tensor(output[0: 4])
output = F.softmax(src, dim=0)
print(output)

v1 = 0.3
v2 = 0.1
x = 0.1
ref = 0
src = torch.Tensor([v1, v2, x, ref])
output = F.softmax(src, dim=0)
print(output)
src = torch.Tensor(output[0: 3])
output = F.softmax(src, dim=0)
print(output)

v1 = 0.3
v2 = 0.5
v3 = 0.1
v4 = 0.3
v5 = 0.4
i1 = 0.5
i2 = 0.8
i3 = 0.1
ref = 0


src = torch.Tensor([v1, v2, v3, v4, v5, i1, i2, i3, ref])
output = F.softmax(src, dim=0)
print(output)
src = torch.Tensor(output[0: 8])
output = F.softmax(src, dim=0)
print(output)
