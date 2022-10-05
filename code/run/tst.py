import numpy as np
import math

a = {1:'a', 2:'c'}
b = {}
# print(len(a))

x = iter(a)
t = len(a)

while t:
    key = next(x)
    b[key] = a[key]
    b[key+2] = 4
    t -= 1

a = [[2,1],[[1,1,1],[2,2,2]]]
b = dict(zip(a[0],a[1]))

# print(a[next(x)])

# for key, value in a.items():
#     # b[key] = value
#     # b[key+2] = 4
#     print(key, value)
    
c = dict(sorted(b.items(), key = lambda x:x[0]))
d = [list(c.keys()), c.values()]
print(d)