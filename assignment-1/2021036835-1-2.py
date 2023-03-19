import numpy as np
import math

#A
m=np.array([i+2 for i in range (25)])
print(m)

#B
m=np.reshape(m,(5,5))
print(m)

#C
for i in range(5) :
    m[i,0]=0
    
print(m)

#D
m=m@m
print(m)

#E
v=m[:1]

a=0
for i in range(5) :
    v[0][i]*=v[0][i]
    a=a+v[0][i]
print(math.sqrt(a))
