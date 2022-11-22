import fmEphys

def calc_LFP(ephys):
    return fmEphys.butterfilt(ephys, lowcut=1, highcut=300, order=5)