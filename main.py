from scipy.io import loadmat
from utils import load_model_from_json
from sampling_generator import sampling_generator
from utils import normalize_data,denormalize_data
from parameter_set import Fs,RR,features
from utils import np, load_csv,get_random_features
import time
import matplotlib.pyplot as plt

filename = 'asl.mat'

annots = loadmat(filename)

offset = 10

size = 40000

imagepath = './images/_'
model_filename = 'pmus_cnn_'+str(size)

def remove_peep(ins_mark,exp_mark,paw): 
    # bool_indexes = ~np.isnan(ins_mark)
    ins_indexes = [i for i in range(len(~np.isnan(ins_mark))) if ~np.isnan(ins_mark)[i]][:-1]
    exp_indexes = [i for i in range(len(~np.isnan(ins_mark))) if ~np.isnan(exp_mark)[i]][1:]
    i = 0
    for  ins, exp in zip(ins_indexes,exp_indexes):
        peep = np.average(paw[ins-5:ins])
        if i == 0: 
            paw[:exp] -= peep
        
        else:
            paw[last_exp:exp] -= peep
        
        last_ins, last_exp = ins,exp
        i+=1

    return paw

# min_flow,max_flow = -230.21053955779516, 291.84186226884987
# min_volume,max_volume = -4.80200698015777, 1625.6115545292896
# min_paw,max_paw = -1.9274642712031753, 15.75437039299909
# min_resistances,max_resistances = 0.004, 0.030009000000008695
# min_capacitances,max_capacitances = 30.01000000000001, 80.00700000006111

# print(min_flow,max_flow)
# print(min_volume,max_volume)
# print(min_paw,max_paw)
# print(min_resistances,max_resistances)
# print(min_capacitances,max_capacitances)
    
    
# model = load_model_from_json(model_filename)
# print(model.summary())


# print(annots['re']) #2 dim
# print(len(annots['re'][1])) #352 tanto 0 quanto 1


# print(annots['flowLmin'])
# print(len(annots['flowLmin']))
if filename is 'asl_assinc.mat':
    aux = [elem[0]*60/1000 for index,elem in enumerate(annots['flow']) if index%offset == 0 ]    
else:
    aux = [elem[0]*60/1000 for index,elem in enumerate(annots['flowLmin']) if index%offset == 0 ]
flow = []
for index in range(len(aux)//901):
    flow.append(aux[index*901:(index+1)*901] )#- np.percentile(np.abs(aux[index*901:(index+1)*901]),15))
# print(len(annots['volint'])) 489998
flow = np.array(flow)    

# get_peep(annots)

# print(min(flow[0,:]),max(flow[0,:]))

#print(annots['exp_mark']) #nan

#print(annots['fs']) #[[512]]

#print(annots['ins']) #Line 1 -> 352

#print(annots['ins_mark']) #nan

# print(annots['paw']) # Good

ins_mark = np.array([elem[0] for index,elem in enumerate(annots['ins_mark'])])
exp_mark = np.array([elem[0] for index,elem in enumerate(annots['exp_mark'])])


_paw = np.array([elem[0] for index,elem in enumerate(annots['paw'])])
_paw = remove_peep(ins_mark,exp_mark,_paw)

aux = np.array([elem for index,elem in enumerate(_paw) if index%offset == 0 ])

paw = []
for index in range(len(aux)//901):
    paw.append(aux[index*901:(index+1)*901])

paw = np.array(paw)

# plt.plot(paw[0])
# plt.show()


# plt.plot(paw[10])
# plt.show()

# plt.plot(paw[20])
# plt.show()

# plt.plot(paw[30])
# plt.show()

# for index in range(len(aux)//901):
#     ins_mark.append(aux[index*901:(index+1)*901] )#- np.percentile(np.abs(aux[index*901:(index+1)*901]),15) )
# ins_mark = np.array(ins_mark)

# print(annots['pmusASL']) # Good
if filename is 'asl_assinc.mat':
    aux = [elem[0] for index,elem in enumerate(annots['pmus']) if index%offset == 0 ]   
else:
    aux = [elem[0] for index,elem in enumerate(annots['pmusASL']) if index%offset == 0 ]
pmus = []
for index in range(len(aux)//901):
    pmus.append(aux[index*901:(index+1)*901] )#- np.percentile(np.abs(aux[index*901:(index+1)*901]),15))
# print(len(annots['volint'])) 489998
pmus = np.array(pmus)

# plt.plot(pmus[0,:])

# print(annots['re']) #2 dim
# print(len(annots['re'][1])) #352 tanto 0 quanto 1

# print(annots['rpm'][0])
# print(len(annots['rpm'][0])) #352 -> 21.98 Same value

# print(len(annots['time'])) #line -> 489998
if filename is 'asl_assinc.mat':
    aux = [elem[0] for index,elem in enumerate(annots['volume']) if index%offset == 0 ]    
else:
    aux = [elem[0] for index,elem in enumerate(annots['volint']) if index%offset == 0 ]
volume = []
for index in range(len(aux)//901):
    volume.append(aux[index*901:(index+1)*901] )#- np.percentile(np.abs(aux[index*901:(index+1)*901]),15))
# print(len(annots['volint'])) 489998
volume = np.array(volume)

# voltacho = [elem for index,elem in enumerate(annots['voltacho']) if index%offset == 0 ]
# print(len(annots['voltacho'])) 489998
# plt.plot(volume[0,:])

# plt.show()

num_examples = flow.shape[0]
num_samples = flow.shape[1]



_flow = np.load('./data/flow'+str(size)+'.npy')
_volume = np.load('./data/volume'+str(size)+'.npy')
_paw = np.load('./data/paw'+str(size)+'.npy')
_resistances = np.load('./data/rins'+str(size)+'.npy')
_capacitances = np.load('./data/capacitances'+str(size)+'.npy')

(min_flow, max_flow, _) = normalize_data(_flow)
(min_volume, max_volume, _) = normalize_data(_volume)
(min_paw, max_paw, _) = normalize_data(_paw)
(min_resistances, max_resistances, _) = normalize_data(_resistances)
(min_capacitances, max_capacitances, _) = normalize_data(_capacitances)

(_, _, flow_norm) = normalize_data(flow, minimum=min_flow, maximum=max_flow)
(_, _, volume_norm) = normalize_data(volume, minimum=min_volume, maximum=max_volume)
(_, _, paw_norm) = normalize_data(paw, minimum=min_paw, maximum=max_paw)

input_data = np.zeros((num_examples, num_samples, 3))
input_data[:, :, 0] = flow_norm
input_data[:, :, 1] = volume_norm
input_data[:, :, 2] = paw_norm


models = [load_model_from_json(model_filename)]

output_pred_test = [model.predict(input_data) for model in models]
output_pred_test = sum(output_pred_test)/len(output_pred_test)

err_r     = []
err_c     = []
err_pmus  = []

# R_hat = np.average([denormalize_data(output_pred_test[i, 0], minimum=min_resistances, maximum=max_resistances) for i in range(num_examples)])
# C_hat = np.average([denormalize_data(output_pred_test[i, 1], minimum= min_capacitances, maximum= max_capacitances) for i in range(num_examples)])

R_hat = denormalize_data(output_pred_test[0, 0], minimum=min_resistances, maximum=max_resistances)
C_hat = denormalize_data(output_pred_test[0, 1], minimum= min_capacitances, maximum= max_capacitances)
alpha = 0.2

rr = min(RR)
fs = max(Fs)
time = np.arange(0, np.floor(180.0 / rr * fs) + 1, 1) / fs

for i in range(num_examples-1):
    # R_hat = alpha*denormalize_data(output_pred_test[i, 0], minimum=min_resistances, maximum=max_resistances) + (1-alpha)*R_hat
    # C_hat = alpha*denormalize_data(output_pred_test[i, 1], minimum= min_capacitances, maximum= max_capacitances) + (1-alpha)*C_hat
    
    R_hat = denormalize_data(output_pred_test[i, 0], minimum=min_resistances, maximum=max_resistances)
    C_hat = denormalize_data(output_pred_test[i, 1], minimum= min_capacitances, maximum= max_capacitances)
    
    # R = denormalize_data(output_data[i, 0], min_resistances, max_resistances)
    # C = denormalize_data(output_data[i, 1], min_capacitances, max_capacitances)
    
    flow = denormalize_data(input_data[i, :, 0], min_flow, max_flow)
    volume = denormalize_data(input_data[i, :, 1], min_volume, max_volume)
    paw = denormalize_data(input_data[i, :, 2], min_paw, max_paw)
    
    print("R:",R_hat)
    print("C:",C_hat)
    # plt.plot(flow)
    # plt.plot(volume)
    # plt.plot(paw)
    # plt.show()
    # raise EOFError()
    
    pmus_hat = paw - (R_hat) * flow *1000.0 / 60.0 - (1 /C_hat) * volume
    
    plt.figure()
    
    plt.plot(time,pmus[i,:])
    plt.plot(time,pmus_hat)
    plt.grid()
    plt.legend(['Real','Rede Neural'])
    plt.ylabel('Pressão muscular (cmH2O)')
    plt.xlabel('Tempo (s)')
    # plt.title('Test case %d' % (i + 1))
    plt.savefig(imagepath +'ppt_pmus_case_test_%d.png' % (i + 1), format='png')
    plt.savefig(imagepath +'ppt_pmus_case_test_%d.svg' % (i + 1), format='svg')
    plt.close()
    err_c.append((C_hat))
    err_r.append((R_hat))
    err_pmus.append(sum((pmus[i,:]-pmus_hat)**2)/len(pmus_hat))

print(sum(err_pmus)/len(err_pmus))

plt.figure()
plt.plot(err_c)
plt.grid()
plt.ylabel('Complacência')
plt.xlabel('Iteração')
# plt.title('Test case %d' % (i + 1))
plt.savefig(imagepath +'ppt_pmus_complacencia.png', format='png')
plt.savefig(imagepath +'ppt_pmus_complacencia.svg', format='svg')
plt.close()

plt.figure()
plt.plot(err_r)
plt.grid()
plt.ylabel('Resistência')
plt.xlabel('Iteração')
# plt.title('Test case %d' % (i + 1))
plt.savefig(imagepath +'ppt_pmus_resistencia.png', format='png')
plt.savefig(imagepath +'ppt_pmus_resistencia.svg', format='svg')
plt.close()


# plt.show()
