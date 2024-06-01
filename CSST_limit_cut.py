import numpy as np
import pandas as pd
import os
import pickle



ex_time = [150,150,150,250,250,250,250,250]   # typical exposure time
ex_num = [1,2,4,1,2,4,8,10]  

magCon = [['z_sn','y_sn'],['i_sn','y_sn'],['i_sn','z_sn'],['r_sn','y_sn'],['r_sn','z_sn'],
          ['r_sn','i_sn'],['g_sn','z_sn']]
limitCon_l = [[25.2,24.4],[25.9,24.4],[25.9,25.2],[26.0,24.4],[26.0,25.2],[26.0,25.9],[26.3,25.2]]
limitCon_h = [[26.4,25.7],[27.0,25.7],[27.0,26.4],[27.2,25.7],[27.2,26.4],[27.2,27.0],[27.5,26.4]]
        

for i in range(6,8):
    for j in range(6):
        dd=open('./df_av/dfTime' + str(ex_num[i]) + '+' + str(ex_time[i]) + 's/df_sn' + str(ex_num[i]) +
            '+'+ str(ex_time[i]) +'s.pkl','rb')
#         dd=open('./df_av/dfTime' + str(ex_num[i])+'+'+str(ex_time[i]) +'s/df_sn' +  str(ex_num[i])+'+'+str(ex_time[i])
#                 + 's_av' + str(j)+ '.pkl','rb')
        df = pickle.load(dd)
        dd.close()
        for k in range(7):
            df2 = df[(df[magCon[k][0]] <= limitCon_h[k][0]) & (df[magCon[k][1]] <= limitCon_h[k][1])]
#  
            dd=open('./df_av/dfTime' + str(ex_num[i]) + '+' +str(ex_time[i]) +'s/df_sn'+str(ex_num[i]) + '+'+str(ex_time[i])+
                    's_' +magCon[k][0][0]+magCon[k][1][0] + '.pkl','wb')
#             dd=open('./model/time%s'%ex_num[i] +'+'+str(ex_time[i]) +'s/mcmc%s'%j + '/tdata%s'%j + '/df_sn' +  
#                     str(ex_num[i])+'+'+str(ex_time[i])+ 's_av'+str(j)+'_'+magCon[k][0][0]+magCon[k][1][0] +'.pkl','wb')
            pickle.dump(df2,dd)
            dd.close()
   

        
    
