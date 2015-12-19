import numpy as np
import scipy.optimize as optimization
import scipy
import math
import csv
import sys
import Module
from Module import *



#So the way this code works is as follows
#1. Defines required functions for 3x permuations, 1x linear function and 1x residuals
#2. Imports data from files- xdata: my_data, ydata: y_train
#3. Loop length of predicors, 1 to 10
#   3.1. Loop candidate predictors
#       3.1.1 Do least squares of candidate predictors with ydata
#       3.1.2 Calculate adjusted R-squared, choosing largest R squared for length of predictor
#   3.2 Save largest Adjusted R-squared with its predicotrs and coefficients
#4. Check graph of Adj R-squared vs No. of predictors to choose best subset of predictors



path='test/'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~  X Data  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#can change the engine to c for larger data. As no column and row names, use integer values
#pred_train=pd.DataFrame.from_csv(path+'predictors_train.csv',index_col=False,header=None)
my_data = np.genfromtxt(path+'predictors_train.csv', delimiter=',',dtype=np.float64)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Y Data  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#y_train=pd.DataFrame.from_csv(path+'returns_train.csv',index_col=False,header=None)
y_train= np.genfromtxt(path+'returns_train.csv', delimiter=',',dtype=np.float64)
rows=len(y_train)



#significance
alpha=0.05

#Put all final predictors with sizes of i
ALL_predictors=[]
#Their corresponding coefficients
ALL_param=[]
#All Largest Adjusted R-squared for final chosen predictors
ALL_La_Adj_R_sq=[]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------i: Number of Predictors-----------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#616665 unique permutations
for i in range(1,11):

    print 'Running %s of 10' %i

    f = open("%i.csv" %i,'w')
    csvwriterf = csv.writer(f, delimiter=',')
    
    #items is an array compositions of 1s and 0s, where their position
    # corresponds to an active column in my_data
    items=np.zeros((20,), dtype=np.int)
    items[0:i]=np.ones((i,), dtype=np.int)
    items=list(items)


    #Empty predictors list
    predictors=np.empty(i, dtype=int, order='C')

    #Use if catching a predictor by smallest sum residuals squared
    smallest_pred=1

    #Largest R-squared
    La_R_sq=0.
    #Largest Adjusted R-squared
    La_Adj_R_sq=0.
    
    #---------------Create candidate item---------------
    for item in perm_unique(items):

        #need to choose the predictors that correspond to 1 in items
        #predictors list becomes a list of columns starting from 0
        j=0
        for col in range(0,len(item)):
            if item[col]==1:
                predictors[j]=col
                j+=1
                
        #Do least squares here!!!!!!!
         #N.B. the x0 ([0.0]*(i+1)) is flattened from a nd array to 1D.

        #For the xdata, using predictors to choose what column data to use
        
        param, cov_x = optimization.leastsq(residuals, [0.0]*(i+1), args=(my_data[:,predictors], y_train))

        SSE      = np.sum(  residuals( param,my_data[:,predictors],y_train )**2 )
        y_mean   = np.mean( y_train )
        SSTO     = np.sum( (y_train-y_mean)**2 )
        R_sq     = 1-SSE/SSTO
        
        #Adj_R_sq can be less than zero
        Adj_R_sq = 1-( (1-R_sq)*(rows-1)/(rows-i-1) )
        if Adj_R_sq<0:
            Adj_R_sq=0

        # Annualised Return, Ann Vol and Sharpe Ratio
        I=func(param,my_data[:,predictors])
        
        returns  = mapl( lambda fr: (fr[0]*fr[1]) , zip(I,y_train))     
        AnnRet   = 252.0*np.mean(returns)*100.0        
        AnnVola  = np.sqrt(252.0)*np.std(returns)*100.0
        SharpeR  = AnnRet / AnnVola
        #fcastRsq = Rsq(I,y_train)
        
        #------------Write data to csv file---------------
        
        #First line:  AnnRet,Sharpe,Volatility,Adj R-sq
        csvwriterf.writerow( (AnnRet, SharpeR, AnnVola, Adj_R_sq) )
        #Second line: Predictors
        csvwriterf.writerow(predictors)
        #Third line:  Coefficients
        csvwriterf.writerow(param)


        
        if Adj_R_sq>La_Adj_R_sq:
            smallest_pred=np.copy(predictors)
            smallest_param=np.copy(param)
            La_R_sq=np.copy(R_sq)
            La_Adj_R_sq=[Adj_R_sq]

    f.close()
    




    
        #The columns of the data frame to use for the least squares have been chosen
        #in predictors
    

    

    #Add predictors to all predictors
    ALL_predictors.append(list(smallest_pred))
    #Add parameters to all parameters
    ALL_param.append(list(smallest_param))
    #print 'Adjusted R-Squared',La_Adj_R_sq
    ALL_La_Adj_R_sq.append(La_Adj_R_sq[0])
    #print 'All Adjusted R-Squared',ALL_La_Adj_R_sq
    print ""





