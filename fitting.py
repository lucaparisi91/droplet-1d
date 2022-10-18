import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import matplotlib
import scipy as sp
from scipy import interpolate
from scipy.optimize import curve_fit

def extrapolateData(data,maxDensity,minDensity,ratio,nH=10,makePlot=False):
    data=data[ data["ratio"]==ratio ]
    dataLow=data[ (data["averageDensity"] <=maxDensity) ]
    x=dataLow["averageDensity"]
    y=(dataLow["energy"] )
    delta=dataLow["deltaenergy"]
    #plt.errorbar(x,y,delta,fmt="o")
    #plt.xlim(0,4)
    #S = lambda x,a,b : 1/(1 + b*(x-a)**2)
    cmap=matplotlib.colormaps["viridis"]
    color=cmap(ratio)
    
    # low density extrapolation
    f = lambda x,b,c,d:    -1  + b*x**(3/2) + c*x**(5/2) + d*x**(6/2)
    params,sigma=curve_fit( f ,x,y,sigma=delta)
    x0=np.linspace(0,np.max(x),num=100000)
    errors=np.sqrt(np.diag(sigma))
    if makePlot:
        plt.plot(x0,f(x0,*params),"--",color=color)
    x0Low=np.linspace(0,np.min(x),num=15)
    y0Low=f(x0Low,*params) 
    errors=np.abs(f(x0Low,*(params + errors)   ) - f(x0Low,*(params - errors)   ))*0 
    
    data2=pd.DataFrame({"ratio" : ratio, "averageDensity":x0Low,"deltaenergy":errors,"energy":y0Low }
                  )
    if makePlot:
        plt.errorbar(data["averageDensity"],data["energy"],data["deltaenergy"],fmt="o",
                 color=color,label="r={}".format(ratio))
        plt.errorbar(data2["averageDensity"],data2["energy"],data2["deltaenergy"],fmt="^",color=color)
    
    # high density extrapolation
    
    dataHigh=data[ (data["averageDensity"] >=minDensity) ]
    x=dataHigh["averageDensity"]
    y=dataHigh["energy"]
    delta=dataHigh["deltaenergy"]
    
    f = lambda x,b,c,d: b*x + c*x**(1/2) +d*x**(3/2)
    params,sigma=curve_fit( f ,x,y,sigma=delta)
    x0=np.linspace(np.min(x),100,num=10000)
    errors=np.sqrt(np.diag(sigma))
    if makePlot:
        plt.plot(x0,f(x0,*params),"--",color=color)
    
    x0=np.linspace(np.max(x),100,num=nH)
    y0=f(x0,*params) 
    errors=y0*0
    dataH=pd.DataFrame({"ratio" : ratio, "averageDensity":x0,"deltaenergy":errors,"energy":y0 }
                  )
    
    if makePlot:
        plt.errorbar(dataH["averageDensity"],dataH["energy"],dataH["deltaenergy"],fmt="s",color=color)

    
    return pd.concat([data,data2,dataH])


def extrapolate(data,makePlot=False):
    dataT9=extrapolateData(data,ratio=0.9,maxDensity=10,minDensity=40,nH=10,makePlot=makePlot)
    dataT6=extrapolateData(data,ratio=0.6,maxDensity=0.9,minDensity=2,nH=200,makePlot=makePlot)
    dataT7=extrapolateData(data,ratio=0.7,maxDensity=2,minDensity=2.5,nH=200,makePlot=makePlot)
    dataT8=extrapolateData(data,ratio=0.8,maxDensity=4,minDensity=4.5,nH=200,makePlot=makePlot)
    dataT75=extrapolateData(data,ratio=0.75,maxDensity=3,minDensity=4,nH=200,makePlot=makePlot)
    dataT65=extrapolateData(data,ratio=0.65,maxDensity=0.9,minDensity=1.2,nH=200,makePlot=makePlot)
    dataT5=extrapolateData(data,ratio=0.5,maxDensity=0.5,minDensity=1.2,nH=200,makePlot=makePlot)
    
    data_extrap=pd.concat([dataT9,dataT6,dataT7,dataT8,dataT75,dataT5,dataT65])
    
    return data_extrap


def fitSpline(data,s=1,weight=False):
    data=data.sort_values(by="averageDensity")
    x=data["averageDensity"]
    y=data["energy"]
    delta=data["deltaenergy"]
    if weight:
        w=np.minimum(1/delta**2, delta*0 + 0.01)
    else:
        w=delta*0+1
    spline=interpolate.UnivariateSpline(x,y=y,s=s,w=w)
    return spline

def fitSplines(data_extrap):
    data=data_extrap.query("ratio==0.9")
    s9=fitSpline(data,s=0.01)
    data=data_extrap.query("ratio==0.6")
    s6=fitSpline(data,s=0.00008)
    data=data_extrap.query("ratio==0.7")
    s7=fitSpline(data,s=3e-4)
    data=data_extrap.query("ratio==0.8")
    s8=fitSpline(data,s=2e-3)
    data=data_extrap.query("ratio==0.75")
    s75=fitSpline(data,s=3e-3)
    data=data_extrap.query("ratio==0.65")
    s65=fitSpline(data,s=3e-4)
    data=data_extrap.query("ratio==0.5")
    s5=fitSpline(data,s=4e-5)
    
    
    # evaluate the splines
    ss=[s9,s8,s7,s6,s75,s65,s5]
    ratios=[0.9,0.8,0.7,0.6,0.75,0.65,0.5]
    
    return ratios,ss


def evaluateSplines(ratios,splines,n0):
    evaluatedSplines=pd.concat( [pd.DataFrame( {"averageDensity": n0,"energy":s(n0), "ratio":r }    ) 
        for  r,s in zip(ratios,splines) ] )
    return evaluatedSplines



def compareSplinesVsQMC(data,data_extrap):
    cmap=matplotlib.colormaps["viridis"]
    for ratio,df in data.groupby("ratio"):
        
        dataQMC=data_extrap[data_extrap["ratio"]==ratio]
        color=cmap(ratio)
        plt.plot(df["averageDensity"],df["energy"],
                 label="r={}".format(ratio),color=color )
        plt.errorbar(dataQMC["averageDensity"],dataQMC["energy"],dataQMC["deltaenergy"],
                     fmt="o",color=color)
    plt.legend()


def sortList(*lists):
    return zip(*sorted(zip(*lists)))
def interpLin(x,x0,y0):
    return y0[0] + (y0[1]-y0[0])/(x0[1]-x0[0]) * (x - x0[0])


class energyInterpolator:
    def __init__(self,qmcData):
        self.data=qmcData
        self.dataExtrap=extrapolate(qmcData)
        self.ratios,self.splines=sortList(*fitSplines(self.dataExtrap) )
        self.ratios=np.array(self.ratios)
        
        
    def __call__(self,density,ratio):
        i=np.argmin( np.abs(ratio - self.ratios ))
        rClose=None
        
        if (ratio > np.max(self.ratios)) or (ratio<np.min(self.ratios)):
            raise ValueError(
                "{} should be containted in ({},{})".format(
                    ratio,np.min(self.ratios),np.max(self.ratios)
                             ) 
                            )
        
        if self.ratios[i] == ratio:
            return self.splines[i](density)
        else:
            if self.ratios[i] > ratio:
                bounds=(i-1,i)
            else:
                bounds=(i,i+1)
        
        s0=self.splines[bounds[0] ]
        s1=self.splines[bounds[1] ]
        r0=self.ratios[bounds[0] ]
        r1=self.ratios[bounds[1] ]

        
        return interpLin( ratio, [r0,r1],[s0(density),s1(density)])