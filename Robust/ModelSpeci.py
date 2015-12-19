import csv
import sys
import numpy as np
import scipy.optimize as optimization
import operator
from operator import truediv
import shutil
import Module
from Module import *

import matplotlib.pyplot as plt




#-------------- Data path for predictors_test.csv--------
path='test/'


#---------------------------------------------------------
#--------------Import data from diagnositics csvs -----------
#----------------------------------------------------------

#Aim here it to reduce the combinations by looking at the figures
# The below loop runs through all the csv files to find 
all_metrics=[]

#----Define min Annual Return and Sharpe Ratio
minAnnRet = 1.3
minSharpe = 3
minForRsq = 0.03




for N in range(1,11):
    
    filename='%s.csv' %N
    try:
        f=open(filename)
    except IOError:
        print '%s not found. Continuing with next file' %filename
        continue


    metrics=[]

    #----------- Certain parameters on every 3rd line in csv files-----
    #Row 1 Metrics: Annual Return, Sharpe Ratio, Annual Volatility, Adj R_sq
    #Row 2 Predictors: number of predictors in the model
    #Row 3 Coefficients: Coefficients of predictors
    
    #Save all data into all_metrics to look like
    #[ [ [AnnRet, SharpeR, AnnVola, Adj_R_sq],[Predictors],[Coefficients] ]     #A model
    #  [ [AnnRet, SharpeR, AnnVola, Adj_R_sq],[Predictors],[Coefficients] ]     #another model
    #                                   ...                                 ]
    
    flag=0
    count=0
    for row in csv.reader(f):
        if count%3==0:
            x=[float(i) for i in row]
            if x[0]> minAnnRet and x[1] > minSharpe and x[3] > minForRsq:
                metrics=[]
                metrics.append(x)
                flag=1        
        elif (count-1)%3==0 and flag==1:
            y=[int(i) for i in row]
            metrics.append([int(i) for i in row])
        elif (count-2)%3==0 and flag==1:
            metrics.append([float(i) for i in row])
            flag=0
            all_metrics.append(metrics)
            
        count+=1
    f.close()
    if not metrics:
        print 'No data above minimum metrics for %s.csv' %N

if not all_metrics:
    print 'No data captured. Try decreasing minimum values for metrics'
    sys.exit()

print ""
print 'Imported subset of train metrics'


#--------------------------------------------------------------
#--------------------- Validation -----------------------------
#--------------------------------------------------------------
# Aim here is to split the trading data into 4 equal parts.
# Training data rows / 4 = testing and trading rows
# and then compare which models work best for all the data splits and test
#  1. Calculate the metrics SharpeR,AnnRet,AnnVola,Adj_R_sq in splits
#  2. For each model, divide each metric by the same metric in trading data.
#  3. For each model, calculate the geometric means of the metrics among all splits
#  4. Sort by a certain metric
#  5. Highest value will be the most robust for that metric. 

print ""
x=len(all_metrics)
print 'Specified No. of Models %s' %x

x_train = np.genfromtxt(path+'predictors_train.csv', delimiter=',',dtype=np.float64)
y_train = np.genfromtxt(path+'returns_train.csv', delimiter=',',dtype=np.float64)
train_rows=len(y_train)

print ""
print 'Conducting Data Splitting and Validation on training data...'

# Data splitting algorithm needs to be edited for non-equal data
data_splits=4
flo_data_splits=float(data_splits)

if train_rows%data_splits==0:
    print '     Data split evenly into %s' %data_splits
    size=train_rows/data_splits
else:
    print '     Data not split evenly'
    print '     Number of rows %s per split' %train_rows
    print '     Split data into %s equal sections' %data_splits
    raise SystemExit(0)


# Where divided metrics will be stored
All_DivMetrics=[]
metrics_test=[]


#For each split in data
for split in range(0,data_splits):

    #Calculate boundaries
    LR = size*split
    UR = size*(split+1)
    print '     Computing split parition %s of %s...' %(split+1,data_splits)


    #for each model
    for i in range(0,x):

        #what are the previously calculated predictors and coeffs
        pred  = all_metrics[i][1]
        param = all_metrics[i][2]
        NoPred=len(pred)

         # For Adj_R_sq
        I        = func(param,x_train[LR:UR,all_metrics[i][1]])

        SSE      = np.sum(  residuals( param , x_train[LR:UR,pred] , y_train[LR:UR] )**2 )
        y_mean   = np.mean( y_train[LR:UR] )
        SSTO     = np.sum( (y_train[LR:UR]-y_mean)**2 )
        R_sq     = 1-SSE/SSTO
        Adj_R_sq = 1-( (1-R_sq)*(size-1)/(size-NoPred-1) )

        # Annualised Return, Ann Vol and Sharpe Ratio

        returns  = mapl( lambda fr: (fr[0]*fr[1]) , zip(I,y_train[LR:UR]))     
        AnnRet   = 252.0*np.mean(returns)*100.0        
        AnnVola  = np.sqrt(252.0)*np.std(returns)*100.0
        SharpeR  = AnnRet / AnnVola

        #fcastRsq = Rsq(I,y_test)
         
        #Divided split train metrics by full train metrics
        a=[AnnRet, SharpeR, AnnVola, Adj_R_sq]
        b=all_metrics[i][0][:-1]

        #Divide first 4 metrics with each other
        # split metrics / all data set metrics
        temp_DivMet=map(truediv, a, b)
                
        if split!=0:
            #Goemetric mean
            All_DivMetrics[i] = map(operator.mul,All_DivMetrics[i],temp_DivMet)
            
            #Arithmetric Mean
            #All_DivMetrics[i] = map(operator.add,All_DivMetrics[i],temp_DivMet)

        #Add new data
        else:
            All_DivMetrics.append(temp_DivMet)


        #If on last run, divide my averaged data points
        if split==data_splits-1:


            #Geometric mean, Root ^(1/data_splits)

            All_DivMetrics[i] = map(lambda m: m**(1/flo_data_splits), All_DivMetrics[i])
            
            #Arithmetric mean
            #All_DivMetrics[i] = map(lambda m: m/data_splits, All_DivMetrics[i])
            
            #ID value corresponding to original all_metrics
            #  Need this as to find model: All_DivMetrics will be sorted
            All_DivMetrics[i].append(i)





#Sorts all divided metrics by adj R^2 in desc order
All_DivMetrics.sort(key=lambda z: z[0])

#Number of final models to use
NoFinalModels=3

#IDs of top predictors and params
IDs=[]

for i in range(0,len(all_metrics)):
    #print All_DivMetrics[i],All_DivMetrics[i][-1]
    IDs.append(All_DivMetrics[i][-1])




strings=['First','Second','Third']
#-----------------------------------------------------------
#------------------------ Forecasts ------------------------------------
#-----------------------------------------------------------------------

data = ['train','test','trade']
x_data = np.genfromtxt(path+'predictors_%s.csv' %data[0], delimiter=',',dtype=np.float64)
y_data = np.genfromtxt(path+'returns_%s.csv' %data[0], delimiter=',',dtype=np.float64)
for i in range(0,3): 
    print 'Model %s' %i 
    pred         = all_metrics[IDs[i]][1]
    param, cov_x = optimization.leastsq(residuals, [0.0]*(len(pred)+1), args=(x_data[:,pred], y_data))
    for j in range(0,len(pred)):
        print 'x_%s' %pred[j],'.','%3.2e' %param[j]
    print '+', '%3.2e' %param[-1]
    print ""




data = ['train','test','trade']
param_save=[]
for section in data:

    x_data = np.genfromtxt( path + 'predictors_%s.csv' %section    , delimiter=',',dtype=np.float64)
    y_data = np.genfromtxt( path + 'returns_%s.csv' %section       , delimiter=',',dtype=np.float64)
    size   = len(y_data)

    print "--------------- %s data forecasts-------------" %section   
    
    print 'Running Top Models on %s data' %section
    
    #-----Running forecasts with models to forecasts_....csv
    for i in range(0,3):
        print path+'forecasts_%s_model%s.csv' %(section,i)
        shutil.copy(path+'returns_%s.csv' %section , path+'returns_%s_model%s.csv' %(section,i))


        pred  = all_metrics[IDs[i]][1]        
        NoPred=len(pred)
        
        print 'Number of Predictors %s' %NoPred

        if section=='train':
            param, cov_x = optimization.leastsq(residuals, [0.0]*(len(pred)+1), args=(x_data[:,pred], y_data))
            param_save.append(param)
        else:
            param=param_save[i]
        
            
        I = func(param,x_data[:,pred])


        np.savetxt(path+'forecasts_%s_model%s.csv' %(section,i),I)

        S        = np.sum(  residuals( param , x_data[:,pred] , y_data )**2 )
        y_mean   = np.mean( y_data )
        SSTO     = np.sum( (y_data-y_mean)**2 )
        R_sq     = 1-S/SSTO
        Adj_R_sq = 1-( (1-R_sq)*(size-1)/(size-NoPred-1) )


        # Annualised Return, Ann Vol and Sharpe Ratio

        returns  = mapl( lambda fr: (fr[0]*fr[1]) , zip(I,y_data))     
        AnnRet   = 252.0*np.mean(returns)*100.0        
        AnnVola  = np.sqrt(252.0)*np.std(returns)*100.0
        SharpeR  = AnnRet / AnnVola

        print 'Sharpe ratio             :', SharpeR
        print 'Annualised return (%)    :', AnnRet
        print 'Annualised volatility (%):', AnnVola
        print 'Adjusted Rsq             :', Adj_R_sq
        print 'MSE                      :', S/float(size)
        print ""
    print ""


