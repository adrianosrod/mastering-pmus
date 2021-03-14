from scipy.io import loadmat
from utils import load_model_from_json
from sampling_generator import sampling_generator
from utils import normalize_data,denormalize_data
from parameter_set import Fs,RR,features
from utils import np, load_csv,get_random_features
import time
import matplotlib.pyplot as plt

filename = 'asl_assinc.mat'

annots = loadmat(filename)

offset = 10

# size = 100e3
# flow = np.load('./data/flow'+str(size)+'.npy').T
# volume = np.load('./data/volume'+str(size)+'.npy').T
# paw = np.load('./data/paw'+str(size)+'.npy').T
# resistances = np.load('./data/rins'+str(size)+'.npy').T
# capacitances = np.load('./data/capacitances'+str(size)+'.npy').T


# (min_flow, max_flow, _) = normalize_data(flow)
# (min_volume, max_volume, _) = normalize_data(volume)
# (min_paw, max_paw, _) = normalize_data(paw)
# (min_resistances, max_resistances, _) = normalize_data(resistances)
# (min_capacitances, max_capacitances, _) = normalize_data(capacitances)

min_flow,max_flow = -230.21053955779516, 291.84186226884987
min_volume,max_volume = -4.80200698015777, 1625.6115545292896
min_paw,max_paw = -1.9274642712031753, 15.75437039299909
min_resistances,max_resistances = 0.004, 0.030009000000008695
min_capacitances,max_capacitances = 30.01000000000001, 80.00700000006111
# print(min_flow,max_flow)
# print(min_volume,max_volume)
# print(min_paw,max_paw)
# print(min_resistances,max_resistances)
# print(min_capacitances,max_capacitances)

imagepath = './images/full_'
model_filename = 'pmus_cnn'
models = []
for i in range(1):
    models.append(load_model_from_json(model_filename+f'_{i+1}'))
    
    
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
    flow.append(aux[index*901:(index+1)*901] - np.percentile(np.abs(aux[index*901:(index+1)*901]),15))
# print(len(annots['volint'])) 489998
flow = np.array(flow)


# print(min(flow[0,:]),max(flow[0,:]))

#print(annots['exp_mark']) #nan

#print(annots['fs']) #[[512]]

#print(annots['ins']) #Line 1 -> 352

#print(annots['ins_mark']) #nan

# print(annots['paw']) # Good
aux = [elem[0] for index,elem in enumerate(annots['paw']) if index%offset == 0 ]
paw = []

for index in range(len(aux)//901):
    paw.append(aux[index*901:(index+1)*901] - np.percentile(np.abs(aux[index*901:(index+1)*901]),15) )
paw = np.array(paw)


# print(annots['pmusASL']) # Good
if filename is 'asl_assinc.mat':
    aux = [elem[0] for index,elem in enumerate(annots['pmus']) if index%offset == 0 ]   
else:
    aux = [elem[0] for index,elem in enumerate(annots['pmusASL']) if index%offset == 0 ]
pmus = []
for index in range(len(aux)//901):
    pmus.append(aux[index*901:(index+1)*901]- np.percentile(np.abs(aux[index*901:(index+1)*901]),15))
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
    volume.append(aux[index*901:(index+1)*901]- np.percentile(np.abs(aux[index*901:(index+1)*901]),15))
# print(len(annots['volint'])) 489998
volume = np.array(volume)

# voltacho = [elem for index,elem in enumerate(annots['voltacho']) if index%offset == 0 ]
# print(len(annots['voltacho'])) 489998
# plt.plot(volume[0,:])

# plt.show()

num_examples = flow.shape[0]
num_samples = flow.shape[1]


(_, _, flow_norm) = normalize_data(flow, minimum=min_flow, maximum=max_flow)
(_, _, volume_norm) = normalize_data(volume, minimum=min_volume, maximum=max_volume)
(_, _, paw_norm) = normalize_data(paw, minimum=min_paw, maximum=max_paw)

input_data = np.zeros((num_examples, num_samples, 3))
input_data[:, :, 0] = flow_norm
input_data[:, :, 1] = volume_norm
input_data[:, :, 2] = paw_norm

output_pred_test = [model.predict(input_data) for model in models]
output_pred_test = sum(output_pred_test)/len(output_pred_test)

err_r     = []
err_c     = []
err_pmus  = []

R_hat = np.average([denormalize_data(output_pred_test[i, 0], minimum=min_resistances, maximum=max_resistances) for i in range(num_examples)])
C_hat = np.average([denormalize_data(output_pred_test[i, 1], minimum= min_capacitances, maximum= max_capacitances) for i in range(num_examples)])
for i in range(num_examples):
    # R_hat = denormalize_data(output_pred_test[i, 0], minimum=min_resistances, maximum=max_resistances)
    # C_hat = denormalize_data(output_pred_test[i, 1], minimum= min_capacitances, maximum= max_capacitances)
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

    
    pmus_hat = paw - np.percentile(np.abs(paw),60) - (R_hat) * flow *1000.0 / 60.0 - (1 /C_hat) * volume
    # pmus_hat = np.where(pmus_hat <= 0, pmus_hat , 0)
    
    plt.figure()
    
    plt.plot(pmus[i,:])
    plt.plot(pmus_hat)
    plt.legend(['Real','Rede Neural'])
    plt.ylabel('Pmus')
    plt.title('Test case %d' % (i + 1))
    plt.savefig(imagepath +'pmus_case_test_%d.png' % (i + 1), format='png')

    # err_c.append((C_hat))
    # err_r.append((R_hat))
    err_pmus.append(sum((pmus[i,:]-pmus_hat)**2)/len(pmus_hat))

print(pmus.shape)
print(sum(err_pmus)/len(err_pmus))

# plt.show()

# plt.plot(err_pmus[0])

from statistics import stdev, mean

# mean_c = mean(err_c)
# std_c  = stdev(err_c)

# print("mean error capacitance: ", mean_c)
# print("std  error capacitance: ", std_c)

# mean_r = mean(err_r)
# std_r  = stdev(err_r)

# print("mean error resistance: ", mean_r)
# print("std  error resistance: ", std_r)

# mean_pmus = mean((mean(err) for err in err_pmus))
# std_pmus  = mean((stdev(err) for err in err_pmus))

# print("mean error pmus: ", mean_pmus)
# print("std  error pmus: ", std_pmus)
