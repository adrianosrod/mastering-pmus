import numpy as np

RESISTENCE_INSP = 'Rins'
RESISTENCE_EXP  = 'Rexp'
CAPACITANCE     = 'C'

Fs = [50.0]           #sample frequency, Hertz (Hz)
Noise = [1e-4, 1e-3, 1e-6] #noise variance over pressure/flow waveforms

Rvent = [0]          #ventilator expiratory valve resistance, cmH2O/(L/s)
Model = ['FOLM'] #respiratory system model
C = np.arange(30.0, 80.01, 0.001)    #respiratory system compliance, (mL/cmH2O)
E2 = [-30, -20, -10, 0, 10, 20, 30]
Rins = np.arange(4.0, 30.01, 0.001)#[7]     #respiratory system inspiratory resistance, cmH2O/(L/s)
Rexp = Rins   #respiratory system  expiratory resistance, cmH2O/(L/s)

PEEP = [0, 5, 15]      #positive end-expiratory pressure, water-centimeters (cmH2O)
SP = [5, 10, 15]        #support pressure (above PEEP) (cmH2O)
Triggertype = ['flow']      #ventilator trigger type
Triggerflow = [2]                          #airflow, (L/min)
Triggerpressure = [-0.5, -1, -2]                 #pressure (cmH2O)
Triggerdelay = [0.05, 0.10]                     #delay time (s)
Triggerarg = Triggerflow
Cycleoff = np.arange(0.10, 0.4 ,0.01)                     #turns off the support, after flow fall below x% of the peak flow
Risetype = ['exp','linear']                    #pressure waveform rises in exponential or linear fashions
Risetime = np.arange(0.15, 0.3,0.01)                    #time (s) to pressure waveform rises from PEEP to SP

RR = np.arange(10.0,35.01,0.1)#[10, 15, 25, 35]   #respiratory rate, respirations per minute (rpm)
Pmustype = ['ingmar', 'linear', 'parexp']        #morphology of the respiratory effort
Pp = np.arange(-15.0, -4.9, 0.001)#[-5]#, -10, -12]    #Pmus negative peak amplitude (cmH2O)
Tp = np.arange(0.3, 0.46, 0.001)#[0.45]#, 0.5]   #Pmus negative peak time (s)
Tf = np.arange(0.6, 0.81, 0.001)#[0.6]     #Pmus finish time (s)

cycles_repetition = [1]                     #repeats n-times the parameter combination

features = {
    'Pmustype': Pmustype,
    'Risetype': Risetype,
    'Triggertype': Triggertype,
    'Model': Model    
}

params = {
    'Fs': Fs,
    'Noise': Noise,
    'Rvent': Rvent,
    'C': C,
    'E2': E2,
    'Rins': Rins,
    'Rexp': Rexp,
    'PEEP': PEEP,
    'SP': SP,
    'Triggerflow':Triggerflow,
    'Triggerpressure': Triggerpressure,
    'Triggerdelay': Triggerdelay,
    'Triggerarg': Triggerarg,
    'Cycleoff': Cycleoff,
    'Risetime': Risetime,
    'RR': RR,
    'Pp': Pp,
    'Tp': Tp,
    'Tf': Tf,
    'cycles_repetition': cycles_repetition
}
