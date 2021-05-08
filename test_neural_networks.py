from utils import load_model_from_json
from sampling_generator import sampling_generator
from utils import normalize_data,denormalize_data
from parameter_set import Fs,RR,features
from utils import np, load_csv,get_random_features
import time
from solve_model import solve_model
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 14})

size = 60000
samples_size = 1000
imagepath = './images/full_'
model_filename = 'pmus__cnn__150_epochs__'+str(size)


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = 100*np.mean(diff/mean)                   # Mean of the difference
    sd        = 100*np.std(diff/mean, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, 100*diff/mean, s = 10,*args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')


rr = min(RR)
fs = max(Fs)
time_ = np.arange(0, np.floor(180.0 / rr * fs) + 1, 1) / fs

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


model = load_model_from_json(model_filename)

input_data = np.load('./data/input_test.npy')
output_data = np.load('./data/output_test.npy')

# flow = flow[:,indexes].T
# volume = volume[:,indexes].T
# paw = paw[:,indexes].T
# resistances = resistances[:,indexes].T
# capacitances = capacitances[:,indexes].T

# num_examples = flow.shape[0]
# num_samples = flow.shape[1]

# (_, _, flow_norm) = normalize_data(flow, minimum=min_flow, maximum=max_flow)
# (_, _, volume_norm) = normalize_data(volume, minimum=min_volume, maximum=max_volume)
# (_, _, paw_norm) = normalize_data(paw, minimum=min_paw, maximum=max_paw)
# (_, _, resistances_norm) = normalize_data(resistances, minimum=min_resistances, maximum=max_resistances)
# (_, _, capacitance_norm) = normalize_data(capacitances, minimum=min_capacitances, maximum=max_capacitances)

# input_data = np.zeros((num_examples, num_samples, 3))
# input_data[:, :, 0] = flow_norm
# input_data[:, :, 1] = volume_norm
# input_data[:, :, 2] = paw_norm
# output_data = np.concatenate((resistances_norm, capacitance_norm), axis=1)

output_pred_test = model.predict(input_data)


plt.rcParams.update({'font.size': 14})

plt.figure()
plt.grid()
# plt.plot(1000*denormalize_data(output_data[0:20, 0], min_resistances, max_resistances))
# plt.plot(1000*denormalize_data(output_pred_test[0:20, 0], min_resistances, max_resistances))
bland_altman_plot(1000*denormalize_data(output_data[:, 0], min_resistances, max_resistances),1000*denormalize_data(output_pred_test[:, 0], min_resistances, max_resistances))
# plt.legend(['Simulator','CNN'])
plt.ylabel('Percentage error (%)')
plt.xlabel('Mean resistance (cmH2O.s/mL)')
plt.xlim(4,30)
plt.ylim(-40,40)
plt.savefig(imagepath + 'resistance.png', format='png')
plt.savefig(imagepath + 'resistance.svg', format='svg')
plt.savefig(imagepath + 'resistance.eps', format='eps')
plt.close()


plt.figure()
plt.grid()
#plt.plot(denormalize_data(output_data[0:20, 1], min_capacitances, max_capacitances))
#plt.plot(denormalize_data(output_pred_test[0:20, 1], min_capacitances, max_capacitances))
bland_altman_plot(denormalize_data(output_data[:, 1], min_capacitances, max_capacitances),denormalize_data(output_pred_test[:, 1], min_capacitances, max_capacitances))
# plt.legend(['Simulator','CNN'])
plt.ylabel('Percentage error (%)')
plt.xlabel('Mean compliance (mL/cmH2O)')
plt.xlim(30,80)
plt.ylim(-40,40)
# plt.title('Capacitance')
plt.savefig(imagepath +'capacitance.png', format='png')
plt.savefig(imagepath +'capacitance.svg', format='svg')
plt.savefig(imagepath +'capacitance.eps', format='eps')

err_r = []
err_c = []
err_pmus = []
err_rc = []

err_pmus_hat = []
err_nmsre = []


plt.rcParams.update({'font.size': 14})

for i in range(samples_size):
    R_hat = denormalize_data(output_pred_test[i, 0], min_resistances, max_resistances)
    C_hat = denormalize_data(output_pred_test[i, 1], min_capacitances, max_capacitances)
    R = denormalize_data(output_data[i, 0], min_resistances, max_resistances)
    C = denormalize_data(output_data[i, 1], min_capacitances, max_capacitances)
    
    flow = denormalize_data(input_data[i, :, 0], min_flow, max_flow)
    volume = denormalize_data(input_data[i, :, 1], min_volume, max_volume)
    paw = denormalize_data(input_data[i, :, 2], min_paw, max_paw)
    
    pmus_hat = paw - (R_hat) * flow *1000.0 / 60.0 - (1 /C_hat) * volume
    pmus     = paw - (R) * flow * 1000.0/ 60.0  - (1 / C) * volume
    
    #err_r.append((R_hat,R))
    #err_c.append((C_hat,C))
    #err_pmus.append((pmus_hat,pmus))
    #err_rc.append((R_hat*C_hat,R*C))
    # plt.figure()
    # plt.grid()
    # plt.plot(time_,pmus)
    # plt.plot(time_,pmus_hat)
    # plt.legend(['Simulator','CNN'])
    # plt.xlabel('Time (s)')
    # plt.xlim(0,18)
    # plt.ylabel('Muscular Pressure (cmH2O)')
    # #plt.title('Test case %d' % (i + 1))
    # plt.savefig(imagepath +'pmus_case_test_%d.png' % (i + 1), format='png')
    # plt.savefig(imagepath +'pmus_case_test_%d.eps' % (i + 1), format='eps')

    err_pmus.extend(pmus)
    err_pmus_hat.extend(pmus_hat)

    #print('test case %d'% (i+1))
    err_nmsre.append(np.sqrt(np.sum((pmus - pmus_hat)**2))/np.sqrt(np.sum((pmus - np.average(pmus))**2)))


# r_squared_error = np.average([(r[0]-r[1])**2 for r in err_r])
# r_error = np.average([r[1] for r in err_r])
# r_hat_error = np.average([r[0] for r in err_r])
# nmse_r = r_squared_error/r_error/r_hat_error


# c_squared_error = np.average([(c[0]-c[1])**2 for c in err_c])
# c_error = np.average([c[1] for c in err_c])
# c_hat_error = np.average([c[0] for c in err_c])
# nmse_c = c_squared_error/c_error/c_hat_error

err_pmus = np.array(err_pmus)
err_pmus_hat = np.array(err_pmus_hat)

nrmse = np.sqrt(np.sum((err_pmus - err_pmus_hat)**2))/np.sqrt(np.sum((err_pmus - np.average(err_pmus))**2))

print(nrmse)
print(np.average(err_nmsre))
print(np.std(err_nmsre))


# rc_squared_error = np.average([(rc[0]-rc[1])**2 for rc in err_rc])
# rc_error = np.average([rc[1] for rc in err_rc])
# rc_hat_error = np.average([rc[0] for rc in err_rc])
# nmse_rc = rc_squared_error/rc_error/rc_hat_error


# plt.figure()
# plt.grid()
# #plt.plot(denormalize_data(output_data[0:20, 1], min_capacitances, max_capacitances))
# #plt.plot(denormalize_data(output_pred_test[0:20, 1], min_capacitances, max_capacitances))
# bland_altman_plot([rc[0] for rc in err_rc],[rc[1] for rc in err_rc])
# # plt.legend(['Simulator','CNN'])
# plt.ylabel('Percentage error (%)')
# plt.xlabel('Mean RC')
# #plt.xlim(30,80)
# # plt.title('Capacitance')
# plt.savefig(imagepath +'rc.png', format='png')
# plt.savefig(imagepath +'rc.svg', format='svg')
# plt.savefig(imagepath +'rc.eps', format='eps')

