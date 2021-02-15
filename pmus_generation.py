from utils import np

def pmus_ingmar(fs, rr, pp, tp, tf):
    """
    Sinusoidal profile
    :param fs: sample frequency
    :param rr: respiratory rate
    :param pp: peak pressure
    :param tp: peak time
    :param tf: end of effort
    :return: pmus profile
    """

    ntp = np.floor(tp * fs)
    ntf = np.floor(tf * fs)
    ntN = np.floor(60.0 * fs / rr)

    pmus1 = np.sin(np.pi * np.arange(0, ntp + 1, 1) / fs / 2.0 / tp)
    pmus2 = np.sin(np.pi / 2.0 / (tf - tp) * (np.arange(ntp + 1, ntf + 1, 1) / fs + tf - 2.0 * tp))
    pmus3 = 0 * np.arange(ntf + 1, ntN + 1, 1) / fs
    pmus = pp * np.concatenate((pmus1, pmus2, pmus3))

    return pmus


def pmus_linear(fs, rr, pp, tp, tf):
    """
    Linear profile
    :param fs: sample frequency
    :param rr: respiratory rate
    :param pp: peak pressure
    :param tp: peak time
    :param tf: end of effort
    :return: pmus profile
    """
    nsamples = np.floor(60.0 / rr * fs)
    time = np.arange(0, nsamples + 1, 1) / fs
    pmus = 0 * time

    for i in range(len(time)):
        if time[i] <= tp:
            pmus[i] = time[i] / tp
        elif time[i] <= tf:
            pmus[i] = (tf - time[i]) / (tf - tp)
        else:
            pmus[i] = 0.0
        pmus[i] = pp * pmus[i]

    return pmus


def pmus_parexp(fs, rr, pp, tp, tf):
    """
    Parabolic-exponential profile
    :param fs: sample frequency
    :param rr: respiratory rate
    :param pp: peak pressure
    :param tp: peak time
    :param tf: end of effort
    :return: pmus profile
    """
    
    ntp = np.floor(tp * fs)
    ntN = np.floor(60.0 / rr * fs)
    taur = abs(tf - tp) / 4.0

    pmus1 = pp * (60.0 * rr - np.arange(0, ntp + 1, 1) / fs) * (np.arange(0, ntp + 1, 1) / fs) / (tp * (60.0 * rr - tp))
    pmus2 = pp * (np.exp(-(np.arange(ntp + 1, ntN + 1, 1) / fs - tp) / taur) - np.exp(-(60.0 * rr - tp) / taur)) / (
                1.0 - np.exp(-(60.0 * rr - tp) / taur))
    pmus = np.concatenate((pmus1, pmus2))

    return pmus


def pmus_passive(fs, rr):
    """
    Passive profile, i.e, no respiratory effort (no spontaneous breathing)
    :param fs: sample frequency
    :param rr: respiratory rate
    :return: pmus profile
    """
    pmus = np.zeros(1, int(np.floor(60.0 / rr * fs) + 1))

    return pmus

#needs coding
def pmus_trapezoidal(fs, rr, ttrap, inspeak, exppeak):
    """
    Trapezoidal profile
    :param fs: sample frequency
    :param rr: respiratory rate
    :param ttrap: time array defining trapezoidal profile
    :param inspeak: negative peak pressure
    :param exppeak: positive peak pressure
    :return: pmus profile
    """
    pmus = np.zeros(1, int(np.floor(60.0 / rr * fs) + 1))
    return pmus


def pmus_profile(fs, rr, pmus_type, pp, tp, tf):
    if pmus_type == 'linear':
        pmus = pmus_linear(fs, rr, pp, tp, tf)
    elif pmus_type == 'ingmar':
        pmus = pmus_ingmar(fs, rr, pp, tp, tf)
    elif pmus_type == 'parexp':
        pmus = pmus_parexp(fs, rr, pp, tp, tf)
    elif pmus_type == 'passive':
        pmus = pmus_passive(fs, rr)
    # elif pmus_type == 'trapezoidal':
    #     pmus = pmus_trapezoidal(fs, rr, ttrap, inspeak, exppeak)
    return pmus