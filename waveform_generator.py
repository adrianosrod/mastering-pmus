from parameter_set import Fs,RR,features
from utils import np, load_csv,get_random_features
import time
from solve_model import solve_model
import os
from sampling_generator import sampling_generator

save_file = True
size = int(100000)
filename = 'test_'+str(size)+'.csv'

if not os.path.exists(filename):
    sampling_generator(size,filename)

header_params , param = load_csv(filename)

num_test_cases = len(param)
header_features, feature = get_random_features(features,num_test_cases)

print(f'Number of respiratory cycles to be simulated: {num_test_cases}')

fs = max(Fs)
rr = min(RR)

print(f'Creating waveforms for fs={fs} / rr={rr}')

num_points = int(np.floor(180.0 / rr * fs)+1)

print("Test cases:", num_test_cases)
print("Number points: ", num_points)

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

# import matplotlib.pyplot as plt

# plt.plot(flow)
# plt.show()
# plt.plot(volume)
# plt.show()
# plt.plot(paw)
# plt.show()

# Save the waveforms in a file
if save_file:
    print(f'Creating file for fs={fs} / rr={rr}')
    np.save('./data/flow'+str(size)+'.npy', flow)
    np.save('./data/volume'+str(size)+'.npy', volume)
    np.save('./data/paw'+str(size)+'.npy', paw)
    np.save('./data/pmus'+str(size)+'.npy', pmus)
    np.save('./data/ins'+str(size)+'.npy', ins)
    np.save('./data/rins'+str(size)+'.npy', resistances_ins)
    np.save('./data/rexp'+str(size)+'.npy', resistances_exp)
    np.save('./data/capacitances'+str(size)+'.npy', capacitances)
