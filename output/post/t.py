import numpy as np

a = [[[1,2,3],['a','b','c']],[[4,5,6],['d','e','f']]]
a = np.array(a)
print(a.shape)
a = a.reshape(1,-1)
a = a.reshape(-1,2,2,3)
print(a)