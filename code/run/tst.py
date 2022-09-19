import numpy as np
import math
# a = [[1,(1.0,2.0,3.0)],[2,(1,2,3)]]
# x_hat = np.zeros((2,3),dtype=np.float)
# x_hat[0] = a[0][1]
# print(x_hat)

# def smoothing_factor(t_e, cutoff):
#     t_e = np.array(t_e)
#     r = 2 * math.pi * cutoff * t_e
#     return r / (r + 1)

# def exponential_smoothing(a, x, x_prev):
#     a = np.array(a)
#     print(a)
#     x = np.array(x)
#     print(x)
#     x_prev = np.array(x_prev)
#     print(x_prev)
#     print(a*x)
#     return a * x + (1 - a) * x_prev
    
# # print(smoothing_factor((1.0,2.0,3.0), 1.0))
# # print((x_hat[0] - [1,1,1]) /2)
# a[0][1] = exponential_smoothing(x_hat[0],[2.1,3.1,4.1],[1.1,2.1,3.1])
# print(dict(a))
a = [1,2,3]
for i in a:
    print(i)