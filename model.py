import pandas as pd
import numpy as np
import multiprocessing
import time
import os
import sys
import emcee
import corner
import pickle
import scipy.stats as stats

path_tdata = ' '
def get_tdata():
    cc=open(path_tdata + 'tdata.pkl','rb')
    tdata=pickle.load(cc)
    cc.close()
    return tdata
  

def redpdf(para,v):
    Av,delta,x,deltac=para   
    if (Av<=0.0) or (delta<=0) or (Av>v[10]):
        return -np.inf
    else:
        color=v[3][0]
        mfred=v[1]
        q=v[3][1]
        phun=v[5]
        pdata=v[8]
        fred=(np.exp(x)/(1+np.exp(x)))**(np.log(mfred)/np.log(0.5))
        kernel=np.random.lognormal(np.log(Av),np.abs(delta),size=len(color)) #kernel=消光值
        red=np.sum([color,kernel*0.124,[deltac*0.124]*len(color)],axis=0)
        color_red,xedge,yedge=np.histogram2d(red,q,bins=[v[9][0],v[9][1]])
        pred0=np.sum([((color_red.flatten())/sum(color_red.flatten()))*fred,phun*(1-fred)],axis=0)
        pred=np.sum([pred0*(1-v[6]),v[7]*v[6]],axis=0)
        
        pmodel=pred
        ins=pmodel==0
        pmodel[ins]=1e-7
        ins2=pdata==0
        pdata[ins2]=1e-7
        return np.sum(pdata-pmodel+pdata*(np.log(pmodel)-np.log(pdata))-np.log(delta)-0.5*np.log(2*np.pi)-np.log(0.3)
                      -((np.log(delta)-np.log(0.3))**2)/(2*(0.3**2))-0.5*np.log(2*np.pi)-np.log(0.035)-((deltac)**4)/(2*(0.035**2)))
#     
    
def run_mc(data,ind):
    nwalkers = 50
    ndim = 4
    p0 = np.random.rand(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, redpdf, args=[data,])
    state = sampler.run_mcmc(p0, 150)
    sampler.reset()
  
    sampler.run_mcmc(state, 15,progress=True)
    samples = sampler.chain[:, :, :].reshape((-1, ndim)
    dd=open('./Sample_Files/'+'data_0_'+str(ind)+'.pkl','wb') #result_path
    pickle.dump(samples,dd)
    dd.close()
    return 0


if __name__=='__main__':    
    ind = int(sys.argv[1])
    tdata = get_tdata()
    run_mc(tdata[ind],ind)
    '''
    start=time.time()
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(4)
    for i in range(10):
        #print(i)
        pool.apply_async(main,args=(tdata[i],))
    pool.close()
    pool.join()
    end=time.time()
    print('Running time: %s Seconds'%(end-start))
    '''
