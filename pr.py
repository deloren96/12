from math import factorial as f
import random
'''
from pr import *
roll(37)
'''
def pr(n,k, b):
    #n = 2**n
    return (f(n)/(f(n-k)*f(k)))/(b**n)
    

def br(n,k, p):
    #n = n+1
    #k = k+1
    print(n,k, k/n)
    return (f(n)/(f(n-k)*f(k)))*p**k*(1-p)**(n-k)

def per(n, g):
    print("i",g-n+g, n, g)
    n = g-n+g
    if n > g:
        return (n-g)/g
    else:
        return (g-n)/g

def roll(j):
    g = 0
    z,r,b = 1,18,18
    for i in range(37, 37+j+1):
    
        #print(i,"z",z, per(z/i, 1/37), "z",r, per(r/i,  (1-1/37)/2), "z",b, per(b/i,  (1-1/37)/2),  )
        #print(i,"z",z, (z+1)/i, "z",r, (r+1)/i, "z",b, (b+1)/i,  )
        print(br(i,z, 1/37), br(i,r, (1-1/37)/2), br(i,b, (1-1/37)/2))
        x = random.randint(0, 36)
        if x == 1:
            g = g+1
        if x in [1, 3 ,5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]:
            r = r+1
        elif x == 0:
            z = z+1
        else:
            b = b+1
    print(g)    
#br(4+1,1+1, 1/2)