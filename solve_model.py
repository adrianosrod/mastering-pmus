from utils import np
from pmus_generation import pmus_profile
from scipy.integrate import odeint


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

    expected_len = int(np.floor(60.0 / 10.0 * fs) + 1)

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
                if paw[i] < sp + peep:
                    paw[i] = (1.0 - np.exp(-(time[i] - time_support) / rise_time * 4.0)) * sp + peep
                if paw[i] >= sp + peep:
                    paw[i] = sp + peep
            elif rise_type == 'linear':
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
                    paw[i] = sp * (np.exp(-(time[i] - time[time_exp-1]) / rise_time * 4.0)) + peep
                if paw[i - 1] <= peep:
                    paw[i] = peep
            elif rise_type == 'linear':
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

    #Adds noise over flow and paw waveforms
    #No noise over volume trace because it is an actual integration of flow trace
    if np.fabs(noise) > 1e-6:
        # flow = flow + np.random.randn(len(flow)) * np.sqrt(noise)
        paw = paw + np.random.randn(len(paw)) * np.sqrt(noise)
        # pmus = pmus + np.random.randn(len(pmus)) * np.sqrt(noise)

    #Drops the first element
    flow = flow[1:] / 1000.0 * 60.0  # converts back to L/min
    volume = volume[1:]
    paw = paw[1:]
    pmus = pmus[1:]
    insex = insex[1:]

    flow = np.concatenate((flow,np.array([0]*(expected_len - len(flow)))))
    volume = np.concatenate((volume,np.array([0]*(expected_len - len(volume)))))
    paw = np.concatenate((paw,np.array([paw[0]]*(expected_len - len(paw)))))
    pmus = np.concatenate((pmus,np.array([pmus[-1]]*(expected_len - len(pmus)))))
    insex = np.concatenate((insex,np.array([insex[0]]*(expected_len - len(insex)))))

    #Consistency check over generated waveforms
    # if not (len(flow) == expected_len and len(volume) == expected_len and
    #         len(paw) == expected_len and len(pmus) == expected_len and len(insex) == expected_len and
    #         volume[-1] < 10.0 and #suitable expiratory time
    #         time_peak_flow > -1 and
    #         time_exp > -1):
    #     if debugmsg:
    #         print(f'{len(flow)}-{len(volume)}-{len(paw)}-{len(pmus)}-{len(insex)}')
    #         print('volume(end)={:.2f}, time_peak_flow={:.2f}, time_exp={:.2}'.format(volume[-1], time_peak_flow, time_exp))

    return flow, volume, paw, pmus, insex, rins,rexp, c

