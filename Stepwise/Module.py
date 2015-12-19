#!/usr/bin/python
# Filename: Module.py
import numpy as np
from numpy import std



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------------------------ Permutations-------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#First 3 functions calculate the unique permutations/combinations
# for all possible predictor variable combinations from my_data

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1


#func is the linear function m1x1+m2x2+m3x3+...mNxN+C
#residuals calculates the residuals from ydata-ymodel
#   returns array
#size n least squares residualstion
# param of size N+1
# param=[m1,m2,m3....mN,C]

def func(param,xdata):
    return np.dot( xdata,param[:-1] )+ param[-1]

def residuals(param,xdata,ydata):
    return ydata-func(param,xdata)

mapl = lambda f,i : list(map(f,i))

def Rsq(fc,r):
    e = mapl(lambda fr: fr[0]-fr[1],zip(fc,r))
    return 1.0 - (std(e)/std(fc))**2

# End of Module.py
