from utils import np
from pmus_generation import pmus_profile
from scipy.integrate import odeint
from scipy import stats
from parameter_set import Fs,RR
import matplotlib.pyplot as plt

def flow_model(y, t, paw, pmus, model, c, e2, r):
    if model == 'FOLM':
        return (paw - pmus - 1.0 / c * y) / r
    elif model == 'VDEM':
        vt = 400  # guessing tidal volume (VT)
        e1 = 1.0 / c
        e2 = e2 / 100  # e2 is in percentage
        if e2 > 0:
            e2true = e1 * e2 / (1 - e2) / vt
        elif e2 < 0:
            e2true = e1 * e2 / (1 + e2) / vt
        elif e2 == 0:
            e2true = 0
        return (paw - pmus - e1 * y - e2true * y ** 2) / r


def solve_model(header_params,params,header_features,features,debugmsg):
    #Extracts each parameter
    fs = params[header_params.index('Fs')]
    rvent = params[header_params.index('Rvent')]
    c = params[header_params.index('C')]
    rins = params[header_params.index('Rins')]
    rexp = rins  # params[4]
    peep = params[header_params.index('PEEP')]
    sp = params[header_params.index('SP')]
    trigger_type = features[header_features.index('Triggertype')]
    trigger_arg = params[header_params.index('Triggerarg')]
    rise_type = features[header_features.index('Risetype')]
    rise_time = params[header_params.index('Risetime')]
    cycle_off = params[header_params.index('Cycleoff')]
    rr = params[header_params.index('RR')]
    pmus_type = features[header_features.index('Pmustype')]
    pp = params[header_params.index('Pp')]
    tp = params[header_params.index('Tp')]
    tf = params[header_params.index('Tf')]
    noise = params[header_params.index('Noise')]
    e2 = params[header_params.index('E2')]
    model = features[header_features.index('Model')]

    expected_len = int(np.floor(180.0 / np.min(RR) * np.max(Fs)) + 1)
    
    #Assings pmus profile
    pmus = pmus_profile(fs, rr, pmus_type, pp, tp, tf)
    pmus = pmus + peep #adjusts PEEP
    pmus = np.concatenate((np.array([0]), pmus)) #sets the first value to zero

    
    #Unit conversion from cmH2O.s/L to cmH2O.s/mL
    rins = rins / 1000.0
    rexp = rexp / 1000.0
    rvent = rvent / 1000.0


    #Generates time, flow, volume, insex and paw waveforms
    time = np.arange(0, np.floor(60.0 / rr * fs) + 1, 1) / fs
    time = np.concatenate((np.array([0]), time))
    flow = np.zeros(len(time))
    volume = np.zeros(len(time))
    insex = np.zeros(len(time))
    paw = np.zeros(len(time)) + peep #adjusts PEEP
    len_time = len(time)

    #Peak flow detection
    peak_flow = flow[0]
    detect_peak_flow = False

    #Support detection
    detect_support = False
    time_support = -1

    #Expiration detection
    detect_exp = False
    time_exp = -1

    if trigger_type == 'flow':
        # units conversion from L/min to mL/s
        trigger_arg = trigger_arg / 60.0 * 1000.0

    for i in range(1, len(time)):
        # period until the respiratory effort beginning
        if (((trigger_type == 'flow' and flow[i] < trigger_arg) or
             (trigger_type == 'pressure' and paw[i] > trigger_arg + peep) or
             (trigger_type == 'delay' and time[i] < trigger_arg)) and
                (not detect_support) and (not detect_exp)):
            paw[i] = peep
            y0 = volume[i - 1]
            tspan = [time[i - 1], time[i]]
            args = (paw[i], pmus[i], model, c, e2, rins)
            sol = odeint(flow_model, y0, tspan, args=args)
            volume[i] = sol[-1]
            flow[i] = flow_model(volume[i], time[i], paw[i], pmus[i], model, c, e2, rins)
            if debugmsg:
                print('volume[i]= {:.2f}, flow[i]= {:.2f}, paw[i]= {:.2f}, waiting'.format(volume[i], flow[i], paw[i]))

            if (((trigger_type == 'flow' and flow[i] >= trigger_arg) or
                 (trigger_type == 'pressure' and paw[i] <= trigger_arg + peep) or
                 (trigger_type == 'delay' and time[i] >= trigger_arg))):
                detect_support = True
                time_support = time[i+1]
                continue

        # detection of inspiratory effort
        # ventilator starts to support the patient
        elif (detect_support and (not detect_exp)):
            if rise_type == 'step':
                paw[i] = sp + peep
            elif rise_type == 'exp':
                rise_type = rise_type if np.random.random() > 0.01 else 'linear'
                if paw[i] < sp + peep:
                    paw[i] = (1.0 - np.exp(-(time[i] - time_support) / rise_time )) * sp + peep
                if paw[i] >= sp + peep:
                    paw[i] = sp + peep
            elif rise_type == 'linear':
                rise_type = rise_type if np.random.random() > 0.01 else 'exp'
                if paw[i] < sp + peep:
                    paw[i] = (time[i] - time_support) / rise_time * sp + peep
                if paw[i] >= sp + peep:
                    paw[i] = sp + peep

            y0 = volume[i - 1]
            tspan = [time[i - 1], time[i]]
            args = (paw[i], pmus[i], model, c, e2, rins)
            sol = odeint(flow_model, y0, tspan, args=args)
            volume[i] = sol[-1]
            flow[i] = flow_model(volume[i], time[i], paw[i], pmus[i], model, c, e2, rins)
            if debugmsg:
                print('volume[i]= {:.2f}, flow[i]= {:.2f}, paw[i]= {:.2f}, supporting'.format(volume[i], flow[i], paw[i]))

            if flow[i] >= flow[i - 1]:
                peak_flow = flow[i]
                detect_peak_flow = False
            elif flow[i] < flow[i - 1]:
                detect_peak_flow = True

            if (flow[i] <= cycle_off * peak_flow) and detect_peak_flow and i<len_time:
                detect_exp = True
                time_exp = i+1    
                try:
                    paw[i + 1] = paw[i]
                except IndexError:
                    pass

        elif detect_exp:
            if rise_type == 'step':
                paw[i] = peep
            elif rise_type == 'exp':
                if paw[i - 1] > peep:
                    paw[i] = sp * (np.exp(-(time[i] - time[time_exp-1]) / rise_time )) + peep
                if paw[i - 1] <= peep:
                    paw[i] = peep
            elif rise_type == 'linear':
                rise_type = rise_type if np.random.random() > 0.01 else 'exp'
                if paw[i - 1] > peep:
                    paw[i] = sp * (1 - (time[i] - time[time_exp-1]) / rise_time) + peep
                if paw[i - 1] <= peep:
                    paw[i] = peep

            y0 = volume[i - 1]
            tspan = [time[i - 1], time[i]]
            args = (paw[i], pmus[i], model, c, e2, rexp + rvent)
            sol = odeint(flow_model, y0, tspan, args=args)
            volume[i] = sol[-1]
            flow[i] = flow_model(volume[i], time[i], paw[i], pmus[i], model, c, e2, rexp + rvent)
            if debugmsg:
                print('volume[i]= {:.2f}, flow[i]= {:.2f}, paw[i]= {:.2f}, exhaling'.format(volume[i], flow[i], paw[i]))

    #Generates InsEx trace
    if time_exp > -1:
        insex = np.concatenate((np.ones(time_exp), np.zeros(len(time) - time_exp)))

    #Drops the first element
    flow = flow[1:] / 1000.0 * 60.0  # converts back to L/min
    volume = volume[1:]
    paw = paw[1:]
    pmus = pmus[1:] - peep #reajust peep again
    insex = insex[1:]

    flow,volume,pmus,insex,paw = generate_cycle(expected_len,flow,volume,pmus,insex,paw,peep=peep)

    # paw = generate_cycle(expected_len,paw,peep=peep)[0]
    
    flow,volume,paw,pmus,insex = generate_noise(noise,flow,volume,paw,pmus,insex)

    # plt.plot(flow)
    # plt.plot(volume)
    # plt.plot(paw)
    # plt.plot(pmus)
    # plt.show()

    return flow, volume, paw, pmus, insex, rins,rexp, c


def generate_cycle(length,*args,peep=None):
    
    startpos  = np.random.randint(0,len(args[0])) 
    delta     = (length - startpos)
    nzeroes   = delta//len(args[0])+1
    
    zpos      = np.random.randint(2,delta//10,size=nzeroes)
    
    res       = [None]*len(args)

    vrandom   = [np.random.uniform(0.95,1.05) if np.random.random() > 0.2 else np.random.uniform(0.1,0.2) for i in range(nzeroes+1)]

    for i in range(len(args)):
        rarr = args[i][-startpos:]
        # mode = stats.mode(args[i])[0]
        mode = peep if (not peep is None) and (i==len(args)-1) else 0
        for j in range(nzeroes):
            aux = vrandom[j+1]*args[i] + (1-vrandom[j+1])*mode
            # aux = 0.0
            # if vrandom[j+1] < 0.9: aux = np.array([ vrandom[j+1]*value if value > mode/vrandom[j+1] else value for value in args[i]])
            # else: aux = np.array([vrandom[j+1]*value if value > mode/vrandom[j+1] else value for value in args[i]])
            rarr = np.concatenate((rarr,np.ones(zpos[j])*mode,aux))

        res[i] = rarr[:length]
    
    return res

def generate_noise(noise,*args):
    
    res = [None]*len(args)
    
    for i in range(len(args)):
        res[i] = args[i] + np.average(args[i])/10*np.random.randn(len(args[i]))*np.sqrt(noise)
    
    return res