import math
import csv
import sys
import numpy as np
import scipy.optimize as optimization
import shutil
import Module
from Module import *


path='test/'
#can change the engine to c for larger data. As no column and row names, use integer values
#pred_train=pd.DataFrame.from_csv(path+'predictors_train.csv',index_col=False,header=None)
x_train = np.genfromtxt(path+'predictors_train.csv', delimiter=',',dtype=np.float64)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Y Data  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#y_train=pd.DataFrame.from_csv(path+'returns_train.csv',index_col=False,header=None)
y_train = np.genfromtxt(path+'returns_train.csv', delimiter=',',dtype=np.float64)

rows=len(y_train)

segments = 4
sep = rows/4


alpha=0.05


#For MSE_FULL, all variables
param, cov_x = optimization.leastsq(residuals, [0.0]*(21), args=(x_train, y_train))
MSE_FULL   =  np.sum(  residuals( param,x_train,y_train )**2 )/float(rows)

all_model_pred=[]
model_pred=[]
for i in range(0,10):
    #Copy list to compare to new one
    #If the same, found best
    old_model=list(model_pred)


    #----------------Forward Seleciton----------------
    max_Adj_R_sq=0
    for cand_pred in range(0,x_train.shape[1]):
        
        #So no duplicate predictors
        if cand_pred in model_pred:
            continue        

        pred=model_pred+[cand_pred]
        param, cov_x = optimization.leastsq(residuals, [0.0]*(len(pred)+1), args=(x_train[:,pred], y_train))

        S        = np.sum(  residuals( param,x_train[:,pred],y_train )**2 )
        y_mean   = np.mean( y_train )
        SSTO     = np.sum( (y_train-y_mean)**2 )
        R_sq     = 1-S/SSTO
        
        #Adj_R_sq can be less than zero
        Adj_R_sq = 1-( (1-R_sq)*(rows-1)/(rows-len(pred)-1) )
        if Adj_R_sq<0:
            Adj_R_sq=0



        # Annualised Return, Ann Vol and Sharpe Ratio
        I=func(param,x_train[:,pred])
        returns  = mapl( lambda fr: (fr[0]*fr[1]) , zip(I,y_train))     
        AnnRet   = 252.0*np.mean(returns)*100.0        
        AnnVola  = np.sqrt(252.0)*np.std(returns)*100.0
        SharpeR  = AnnRet / AnnVola

        
        if Adj_R_sq>max_Adj_R_sq:
            best_pred=[cand_pred]
            max_Adj_R_sq=Adj_R_sq



    model_pred=model_pred+list(best_pred)


    #------------------Backwards elimination---------------
    best_rem=[]
    for rem_pred in model_pred:

        pred=list(set(model_pred) - set([rem_pred]))

        param, cov_x = optimization.leastsq(residuals, [0.0]*(len(pred)+1), args=(x_train[:,pred], y_train))

        S        = np.sum(  residuals( param,x_train[:,pred],y_train )**2 )
        y_mean   = np.mean( y_train )
        SSTO     = np.sum( (y_train-y_mean)**2 )
        R_sq     = 1-S/SSTO
        #Adj_R_sq can be less than zero
        Adj_R_sq = 1-( (1-R_sq)*(rows-1)/(rows-len(pred)-1) )
        if Adj_R_sq<0:
            Adj_R_sq=0

        MSE=S/float(rows)
        Cp       = S/MSE_FULL - (rows-2*len(pred))
        

        # Annualised Return, Ann Vol and Sharpe Ratio
        I=func(param,x_train[:,pred])
        returns  = mapl( lambda fr: (fr[0]*fr[1]) , zip(I,y_train))     
        AnnRet   = 252.0*np.mean(returns)*100.0        
        AnnVola  = np.sqrt(252.0)*np.std(returns)*100.0
        SharpeR  = AnnRet / AnnVola



        if Adj_R_sq>max_Adj_R_sq:
            best_rem=[rem_pred]
            max_Adj_R_sq=Adj_R_sq


    model_pred=list(set(model_pred) - set(best_rem))

    if set(old_model)==set(model_pred):
        print 'No Change in Models'
        break


    
    all_model_pred.append(model_pred)




#-------------------Finding smallest Mallow's Cp-------------
Cp_list=[]
for pred in all_model_pred:

    param, cov_x = optimization.leastsq(residuals, [0.0]*(len(pred)+1), args=(x_train[:,pred], y_train))

    S        = np.sum(  residuals( param,x_train[:,pred],y_train )**2 )
    MSE      = S/float(rows)
    Cp       = S/MSE_FULL - (rows-2*len(pred))
    Cp_list.append(Cp)


# Min
mins1   = min(Cp_list)
loc1    = Cp_list.index(mins1)
# Second min
mins2   = min(n for n in Cp_list if n!=mins1)
loc2    = Cp_list.index(mins2)
# Third Min
mins3   = min(n for n in Cp_list if (n!=mins1 and n!=mins2))
loc3    = Cp_list.index(mins3)

locs=[loc1,loc2,loc3]

#Create x axis of predictor lengths for plot
x = []
for i in range(0,len(all_model_pred)):
    x.append(len(all_model_pred[i]))



data = ['train','test','trade']
#------------------Forecasts----------------------
#  Plots models to screen and forecasts to files
print ""
print ""
print "--------------- Models -------------"
print ""


x_data = np.genfromtxt(path+'predictors_%s.csv' %data[0], delimiter=',',dtype=np.float64)
y_data = np.genfromtxt(path+'returns_%s.csv' %data[0], delimiter=',',dtype=np.float64)
for i in range(0,3): 
    print 'Model %s' %i 
    pred         = all_model_pred[locs[i]]
    param, cov_x = optimization.leastsq(residuals, [0.0]*(len(pred)+1), args=(x_data[:,pred], y_data))
    for j in range(0,len(all_model_pred[locs[i]])):
        print 'x_%s' %all_model_pred[locs[i]][j],'.','%3.2e' %param[j]
    print '+', '%3.2e' %param[-1]
    print ""


param_save=[]
for section in data:

    x_data = np.genfromtxt(path+'predictors_%s.csv' %section, delimiter=',',dtype=np.float64)
    y_data = np.genfromtxt(path+'returns_%s.csv' %section, delimiter=',',dtype=np.float64)
    size=len(y_data)

    print "--------------- %s data forecasts-------------" %section   

    
    
    print 'Running Top Models on %s data' %section
    #-----Running forecasts with models to forecasts_....csv
    for i in range(0,3):
        print path+'forecasts_%s_model%s.csv' %(section,i)
        shutil.copy(path+'returns_%s.csv' %section , path+'returns_%s_model%s.csv' %(section,i))

        pred  = all_model_pred[locs[i]]
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





