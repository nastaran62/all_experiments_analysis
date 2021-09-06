import numpy as np
import heartpy as hp

def get_ppg_components(data, sampling_rate, window_size=20, overlap=19):
    '''
    Extracts HR, HRV and breathing rate from PPG

    @param np.array ppg_data: PPG data
    @param int sampling_rate: PPG sampling rate

    @keyword int window_length: Length of sliding window for measurment in seconds
    @keyword float overlap: Amount of overlap between two windows in seconds

    @rtype: dict(str: numpy.array)
    @note: dict.keys = ["hr", "hrv", "breathing_rate"]

    @return a dictionary of PPG components
    '''

    wd, m = hp.process_segmentwise(data,
                                   sample_rate=sampling_rate,
                                   segment_width=window_size,
                                   segment_overlap=overlap/window_size)
    hr = None
    hrv = None
    breathing_rate = None

    if 'bpm' in m.keys():
        hr = np.array(m['bpm'])
    
    if 'sdsd' in m.keys():
        hrv = np.array(m['sdsd'])

    if 'breathingrate' in m.keys():
        breathing_rate = np.array(m['breathingrate'])
    return hr, hrv, breathing_rate

