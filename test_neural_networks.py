from utils import load_model_from_json
from sampling_generator import sampling_generator
from utils import normalize_data,denormalize_data
from parameter_set import Fs,RR,features
from utils import np, load_csv,get_random_features
import time
from solve_model import solve_model
import matplotlib.pyplot as plt

size = 60000.0
flow = np.load('./data/flow'+str(size)+'.npy')
volume = np.load('./data/volume'+str(size)+'.npy')
paw = np.load('./data/paw'+str(size)+'.npy')
resistances = np.load('./data/rins'+str(size)+'.npy')
capacitances = np.load('./data/capacitances'+str(size)+'.npy')

(min_flow, max_flow, _) = normalize_data(flow)
(min_volume, max_volume, _) = normalize_data(volume)
(min_paw, max_paw, _) = normalize_data(paw)
(min_resistances, max_resistances, _) = normalize_data(resistances)
(min_capacitances, max_capacitances, _) = normalize_data(capacitances)

# min_flow,max_flow = -230.21053955779516, 291.84186226884987
# min_volume,max_volume = -4.80200698015777, 1625.6115545292896
# min_paw,max_paw = -1.9274642712031753, 15.75437039299909
# min_resistances,max_resistances = 0.004, 0.030009000000008695
# min_capacitances,max_capacitances = 30.01000000000001, 80.00700000006111


imagepath = './images/full_'
model_filename = 'pmus_cnn_ASL_sincrono'
model = load_model_from_json(model_filename)
# print(model.summary())
path_samples = 'test_21.csv'
# sampling_generator(100,path_samples)

header_params , param = load_csv(path_samples)

num_test_cases = len(param)
header_features, feature = get_random_features(features,num_test_cases)

print(f'Number of respiratory cycles to be simulated: {num_test_cases}')

fs = Fs[0]
rr = RR[0]

print(f'Creating waveforms for fs={fs} / rr={rr}')

num_points = int(np.floor(180.0 / np.min(RR) * np.max(Fs)) + 1)

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
plt.grid()
plt.plot(denormalize_data(output_data[0:20, 0], min_resistances, max_resistances))
plt.plot(denormalize_data(output_pred_test[0:20, 0], min_resistances, max_resistances))
plt.legend(['Real','Rede Neural'])
plt.ylabel('Resistência')
plt.xlabel('Iterações')
plt.savefig(imagepath + 'resistance.png', format='png')
plt.savefig(imagepath + 'resistance.svg', format='svg')
plt.close()


plt.figure()
plt.grid()
plt.plot(denormalize_data(output_data[0:20, 1], min_capacitances, max_capacitances))
plt.plot(denormalize_data(output_pred_test[0:20, 1], min_capacitances, max_capacitances))
plt.legend(['Real','Rede Neural'])
plt.ylabel('Complacência')
plt.xlabel('Iterações')
# plt.title('Capacitance')
plt.savefig(imagepath +'capacitance.png', format='png')
plt.savefig(imagepath +'capacitance.svg', format='svg')

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

