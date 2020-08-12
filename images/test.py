import numpy as np
import torch

his = torch.randn(3,3)
b = np.array([[1, 2, 3],
            [4, 1, 4],
            [2, 6, 0]])

a = np.diag(his)/np.diag(b)
a1 = np.nanmean(a)
print(his)
print('a=',a)
print('a1= ',a1)