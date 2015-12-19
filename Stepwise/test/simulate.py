import csv
import sys
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('period')
parser.add_argument('-s','--suffix',metavar='SUFFIX', action='store',required=False,default=None,help='Forecast filename suffix.')

args = parser.parse_args()

if args.period not in ['train','test','trade']:
    period = 'test'
else:
    period = args.period

mapl = lambda f,i : list(map(f,i)) # python3 compatibility

try:
    from numpy import std, loadtxt as loadtxt_, mean, sqrt
    def loadtxt(*a,**k):
        data = loadtxt_(*a,**k)
        if data.ndim == 1:
            data.shape = data.size, 1
        return data.tolist()
except ImportError:
    # if user has no numpy
    from math import sqrt
    def loadtxt(filename,delimiter):
        with open(filename,'r') as f:
            rows = mapl(lambda r: mapl(float,r),csv.reader(f))
        return rows

    def mean(arr):
        return sum(arr)/float(len(arr))

    def std(arr):
        m = mean(arr)
        return sqrt(mean(mapl(lambda x: (x-m)**2,arr)))

def Rsq(fc,r):
    e = mapl(lambda fr: fr[0]-fr[1],zip(fc,r))
    return 1.0 - (std(e)/std(fc))**2

def maxdrawdown(returns):
    cr = mapl(lambda i: sum(returns[:i]), range(len(returns)))
    dd = mapl(lambda i: max(cr[:i]+[0])-cr[i],range(len(cr)))
    return max(dd)

if args.suffix is not None:
    responseFile = 'returns_%s_%s.csv' % (period,args.suffix)
    forecastFile = 'forecasts_%s_%s.csv' % (period,args.suffix)
else:
    responseFile = 'returns_%s.csv' % period
    forecastFile = 'forecasts_%s.csv' % period

responses = mapl(lambda r: r[0],loadtxt(responseFile, delimiter=','))
forecasts = mapl(lambda l: l[0],loadtxt(forecastFile, delimiter=','))

returns = mapl(lambda fr: (fr[0]*fr[1]), zip(forecasts,responses))
drawdown = maxdrawdown(returns)
fcastRsq = Rsq(forecasts,responses)

annReturns = 252.0*mean(returns)*100.0
annVolatility = sqrt(252.0)*std(returns)*100.0
print('------------------------------------')
print('Simulated trading strategy results')
print('------------------------------------')
print('Sharpe ratio              : %-6.2f' % (annReturns/annVolatility))
print('Annualised return (%%)     : %-6.1f' % annReturns)
print('Annualised volatility (%%) : %-6.1f' % annVolatility)
print('Max Drawdown (%%)          : %-6.1f' % (drawdown*100.0))
print('Forecast R^2              : %-6.2f' % fcastRsq)
print('------------------------------------')
