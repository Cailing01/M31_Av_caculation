import os 
import multiprocessing
import time 

n_poll, n_tdata = 5, 600

def main(index):
    print(index)
    st = 'python3 -W ignore test0.py '+str(index)
    os.system(st)
    return 0
if __name__=='__main__':  
#     time_start=time.time()
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()#note
    for i in range(n_tdata):
          pool.apply_async(main,args=(i,))
    pool.close()
    pool.join()
    
#     time_end=time.time()
#     print (time_end-time_start)
