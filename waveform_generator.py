from parameter_set import Fs,RR,features
from utils import np, load_csv,get_random_features
import time
from solve_model import solve_model
import os
from sampling_generator import sampling_generator
import matplotlib.pyplot as plt

save_file = True
size = 60000
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
    if i % 500 == 0:
        print('%d/%d' % (i, num_test_cases))
    (flow[:, i], volume[:, i], paw[:, i], pmus[:, i], ins[:, i], rins,rexp, c) = solve_model(header_params,param[i], header_features,feature[i],'')
    resistances_ins[0, i] = rins
    resistances_exp[0,i] = rexp
    capacitances[0, i] = c

print(f'Elapsed time for solving test cases: {time.time() - t}')


# time = np.arange(0, len(flow), 1) / Fs
# path = './images/ppt/'

# for index in range(size):

#     plt.figure()
#     plt.grid()
#     plt.plot(time,volume[:,index])
#     plt.xlim(0,18)
#     plt.ylim(0,550)
#     plt.xlabel('Tempo ($s$)',fontsize=14)
#     plt.ylabel('Volume ($mL$)',fontsize=14)
#     plt.tight_layout()
#     plt.savefig(path+str(index)+'_volume.eps',format='eps')
#     plt.savefig(path+str(index)+'_volume.svg',format='svg')
#     plt.savefig(path+str(index)+'_volume.png',format='png')
#     plt.close()

#     plt.figure()
#     plt.grid()
#     plt.plot(time,flow[:,index]*1000/60)
#     plt.xlim(0,18)
#     plt.ylim(-1000,1550)
#     plt.xlabel('Tempo ($s$)',fontsize=14)
#     plt.ylabel('Fluxo ($mL/s$)',fontsize=14)
#     plt.tight_layout()
#     plt.savefig(path+str(index)+'_fluxo.eps',format='eps')
#     plt.savefig(path+str(index)+'_fluxo.svg',format='svg')
#     plt.savefig(path+str(index)+'_fluxo.png',format='png')
#     plt.close()


#     plt.figure()
#     plt.grid()
#     plt.plot(time,paw[:,index])
#     plt.xlim(0,18)
#     plt.ylim(0,25)
#     plt.xlabel('Tempo ($s$)',fontsize=14)
#     plt.ylabel('Pressão Respirador ($cmH2O$)',fontsize=14)
#     plt.tight_layout()
#     plt.savefig(path+str(index)+'_pressao.eps',format='eps')
#     plt.savefig(path+str(index)+'_pressao.svg',format='svg')
#     plt.savefig(path+str(index)+'_pressao.png',format='png')
#     plt.close()


#     plt.figure()
#     plt.grid()
#     plt.plot(time,pmus[:,index])
#     plt.xlim(0,18)
#     plt.ylim(-18,2)
#     plt.xlabel('Tempo ($s$)',fontsize=14)
#     plt.ylabel('Pressão Muscular ($cmH2O$)',fontsize=14)
#     plt.tight_layout()
#     plt.savefig(path+str(index)+'_pmus.eps',format='eps')
#     plt.savefig(path+str(index)+'_pmus.svg',format='svg')
#     plt.savefig(path+str(index)+'_pmus.png',format='png')
#     plt.close()

#     plt.figure()
#     plt.grid()
#     plt.plot(time,volume[:,index]/10)
#     plt.plot(time,flow[:,index])
#     plt.plot(time,paw[:,index])
#     plt.plot(time,pmus[:,index])
#     plt.xlim(0,18)
#     # plt.ylim(-18,2)
#     plt.xlabel('Tempo ($s$)',fontsize=14)
#     plt.legend(['Volume (cL)','Fluxo (L/min)','Pressão (cmH2O)','Pressão Muscular (cmH2O)'])
#     # plt.ylabel('Pressão Muscular ($cmH2O$)')
#     plt.tight_layout()
#     plt.savefig(path+str(index)+'_compilado.eps',format='eps')
#     plt.savefig(path+str(index)+'_compilado.svg',format='svg')
#     plt.savefig(path+str(index)+'_compilado.png',format='png')
#     plt.close()

    


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
