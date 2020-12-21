from utils import load_model_from_json
from sampling_generator import sampling_generator
from utils import normalize_data,denormalize_data
from parameter_set import Fs,RR,features
from utils import np, load_csv,get_random_features
import time
from solve_model import solve_model
import matplotlib.pyplot as plt

flow = np.load('flow200000.npy')
volume = np.load('volume200000.npy')
paw = np.load('paw200000.npy')
resistances = np.load('rins200000.npy')
capacitances = np.load('capacitances200000.npy')

(min_flow, max_flow, _) = normalize_data(flow)
(min_volume, max_volume, _) = normalize_data(volume)
(min_paw, max_paw, _) = normalize_data(paw)
(min_resistances, max_resistances, _) = normalize_data(resistances)
(min_capacitances, max_capacitances, _) = normalize_data(capacitances)

imagepath = './images/full_'
model_filename = 'pmus_cnn'
model = load_model_from_json(model_filename)
print(model.summary())
path_samples = 'test_small_100.csv'
sampling_generator(100,path_samples)

header_params , param = load_csv(path_samples)

num_test_cases = len(param)
header_features, feature = get_random_features(features,num_test_cases)

print(f'Number of respiratory cycles to be simulated: {num_test_cases}')

fs = Fs[0]
rr = RR[0]

print(f'Creating waveforms for fs={fs} / rr={rr}')

num_points = int(np.floor(60.0 / rr * fs) + 1)

# Target waveforms
flow = np.zeros((num_points, num_test_cases))
volume = np.zeros((num_points, num_test_cases))
paw = np.zeros((num_points, num_test_cases))
pmus = np.zeros((num_points, num_test_cases))
ins = np.zeros((num_points, num_test_cases))
resistances_ins = np.zeros((1, num_test_cases))
resistances_exp = np.zeros((1, num_test_cases))
capacitances = np.zeros((1, num_test_cases))

t = time.time()
for i in range(num_test_cases):
    if i % 1000 == 0:
        print('%d/%d' % (i, num_test_cases))
    (flow[:, i], volume[:, i], paw[:, i], pmus[:, i], ins[:, i], rins,rexp, c) = solve_model(header_params,param[i], header_features,feature[i],'')
    resistances_ins[0, i] = rins
    resistances_exp[0,i] = rexp
    capacitances[0, i] = c

print(f'Elapsed time for solving test cases: {time.time() - t}')


flow = flow.T
volume = volume.T
paw = paw.T
resistances = resistances_ins.T
capacitances = capacitances.T

num_examples = flow.shape[0]
num_samples = flow.shape[1]

(_, _, flow_norm) = normalize_data(flow, minimum=min_flow, maximum=max_flow)
(_, _, volume_norm) = normalize_data(volume, minimum=min_volume, maximum=max_volume)
(_, _, paw_norm) = normalize_data(paw, minimum=min_paw, maximum=max_paw)
(_, _, resistances_norm) = normalize_data(resistances, minimum=min_resistances, maximum=max_resistances)
(_, _, capacitance_norm) = normalize_data(capacitances, minimum=min_capacitances, maximum=max_capacitances)

input_data = np.zeros((num_examples, num_samples, 3))
input_data[:, :, 0] = flow_norm
input_data[:, :, 1] = volume_norm
input_data[:, :, 2] = paw_norm
output_data = np.concatenate((resistances_norm, capacitance_norm), axis=1)

output_pred_test = model.predict(input_data)

plt.figure()
plt.plot(denormalize_data(output_data[0:20, 0], min_resistances, max_resistances))
plt.plot(denormalize_data(output_pred_test[0:20, 0], min_resistances, max_resistances))
plt.legend(['Real','Rede Neural'])
plt.ylabel('R')
plt.title('Resistance')
plt.savefig(imagepath + 'resistance.png', format='png')



plt.figure()
plt.plot(denormalize_data(output_data[0:20, 1], min_capacitances, max_capacitances))
plt.plot(denormalize_data(output_pred_test[0:20, 1], min_capacitances, max_capacitances))
plt.legend(['Real','Rede Neural'])
plt.ylabel('C')
plt.title('Capacitance')
plt.savefig(imagepath +'capacitance.png', format='png')

err_r     = []
err_c     = []
err_pmus  = []

for i in range(20):
    R_hat = denormalize_data(output_pred_test[i, 0], min_resistances, max_resistances)
    C_hat = denormalize_data(output_pred_test[i, 1], min_capacitances, max_capacitances)
    R = denormalize_data(output_data[i, 0], min_resistances, max_resistances)
    C = denormalize_data(output_data[i, 1], min_capacitances, max_capacitances)
    
    flow = denormalize_data(input_data[i, :, 0], min_flow, max_flow)
    volume = denormalize_data(input_data[i, :, 1], min_volume, max_volume)
    paw = denormalize_data(input_data[i, :, 2], min_paw, max_paw)
    
    pmus_hat = paw - (R_hat) * flow *1000.0 / 60.0 - (1 /C_hat) * volume
    pmus     = paw - (R) * flow * 1000.0/ 60.0  - (1 / C) * volume
    
    plt.figure()
    plt.plot(pmus)
    plt.plot(pmus_hat)
    plt.legend(['Real','Rede Neural'])
    plt.ylabel('Pmus')
    plt.title('Test case %d' % (i + 1))
    plt.savefig(imagepath +'pmus_case_test_%d.png' % (i + 1), format='png')

    err_c.append((C-C_hat)**2)
    err_r.append((R-R_hat)**2)
    err_pmus.append((pmus-pmus_hat)**2)


from statistics import stdev, mean

mean_c = mean(err_c)
std_c  = stdev(err_c)

print("mean error capacitance: ", mean_c)
print("std  error capacitance: ", std_c)

mean_r = mean(err_r)
std_r  = stdev(err_r)

print("mean error resistance: ", mean_r)
print("std  error resistance: ", std_r)

mean_pmus = mean((mean(err) for err in err_pmus))
std_pmus  = mean((stdev(err) for err in err_pmus))

print("mean error pmus: ", mean_pmus)
print("std  error pmus: ", std_pmus)


# plt.figure()
# plt.plot(rc)
# plt.plot(rc_hat)    
# plt.show()

