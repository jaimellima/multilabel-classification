

from re import X
import numpy as np


import numpy as np
arr = np.array([[1, 2, 3], 
                [0, 1, 0], 
                [7, 0, 2]
                ])
teste = np.argwhere(arr == 0)
print(teste[:,0])
print(teste[:,1])