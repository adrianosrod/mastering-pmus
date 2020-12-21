from parameter_set import params,RESISTENCE_INSP,RESISTENCE_EXP,CAPACITANCE
from utils import save_csv,is_in_range,np,pd,choice

def sampling_generator(size, path=''):    
    samples = []
    len_params = len(params)
    map_params = list(params.keys())
    
    while len(samples) < size:
        
        vec = [None]*len_params
        
        for param, arr in params.items():
            vec[map_params.index(param)] = choice(arr) 
        
        if is_valid(vec,map_params):
            samples.append(vec)  
    
    if path:
        save_csv(samples,map_params,path)    
    
    return samples


def is_valid(arr,map_params):
    resp = False

    if is_in_range(arr[map_params.index(RESISTENCE_INSP)],4.0,10.1) and is_in_range(arr[map_params.index(RESISTENCE_EXP)],4.0,10.0):
        resp = is_in_range(arr[map_params.index(CAPACITANCE)],60.01,80.01)
    elif is_in_range(arr[map_params.index(RESISTENCE_INSP)],10.01,20.01) and is_in_range(arr[map_params.index(RESISTENCE_EXP)],10.01,20.01):
        resp = is_in_range(arr[map_params.index(CAPACITANCE)],40.01,60.01)
    elif is_in_range(arr[map_params.index(RESISTENCE_INSP)],20.01,30.01) and is_in_range(arr[map_params.index(RESISTENCE_EXP)],20.01,30.01):
        resp = is_in_range(arr[map_params.index(CAPACITANCE)],30.01,40.01)
    
    return resp
