import numpy as np
import scipy

def prop_neg_derivatives(data):
    feature = (data < 0).sum()/np.product(data.shape)
    return [feature]


def get_local_maxima(data):
    '''
    Reterns local maximums
    '''
    return [data[i] for i in scipy.signal.argrelextrema(data, np.greater)[0]]


def get_local_minima(data):
    '''
    Reterns local minimums
    '''
    return [data[i] for i in scipy.signal.argrelextrema(data, np.less)[0]]


def get_frequency_peak(data):
    '''
    Reterns frequency of occurrence of local extremes
    '''
    local_maxima = data.get_local_maxima()
    local_minima = data.get_local_minima()

    freq_extremes = len(local_maxima) + len(local_minima)

    return [freq_extremes]


def get_max_amp_peak(data):
    '''
    Returns the highest value of the determined maximums if it exists. Otherwise it returns zero
    '''
    local_maxima = list(data.get_local_maxima()) + [0]
    return [max(local_maxima)]


def get_var_amp_peak(data):
    '''
    Returns variance of amplitude values calculated for local extremes
    '''
    amplitude_of_local_maxima = np.absolute(data.get_local_maxima())
    amplitude_of_local_minima = np.absolute(data.get_local_minima())
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    variance = np.var(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

    return [variance]


def std_amp_peak(data):
    '''
    Returns the standard deviation calculated for local extremes
    '''

    local_extremes = list(data.get_local_maxima()) + \
                        list(data.get_local_minima())
    if len(local_extremes) == 0:
        return [0]
    return [np.std(local_extremes)]


def skewness_amp_peak(data):
    '''
    Retruns the skewness calculated for amplitude of local extremes
    '''
    amplitude_of_local_maxima = np.absolute(data.get_local_maxima())
    amplitude_of_local_minima = np.absolute(data.get_local_minima())
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    skewness = scipy.stats.skew(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

    return [skewness]


def kurtosis_amp_peak(data):
    '''
    Retruns the kurtosis calculated for amplitude of local extremes
    '''
    amplitude_of_local_maxima = np.absolute(data.get_local_maxima())
    amplitude_of_local_minima = np.absolute(data.get_local_minima())
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    kurtosis = scipy.stats.kurtosis(list(amplitude_of_local_maxima) +
                                    list(amplitude_of_local_minima))

    return [kurtosis]


def max_abs_amp_peak(data):
    '''
    Retruns the kurtosis calculated for amplitude of local extremes
    '''
    amplitude_of_local_maxima = np.absolute(data.get_local_maxima())
    amplitude_of_local_minima = np.absolute(data.get_local_minima())
    if len(amplitude_of_local_maxima) + len(amplitude_of_local_minima) == 0:
        return [0]
    max_val = max(list(amplitude_of_local_maxima) + list(amplitude_of_local_minima))

    return [max_val]


def variance(data):
    '''
    Returns the variance of the data
    '''
    var = np.var(data)

    return [var]


def standard_deviation(data):
    '''
    Returns the standard deviation of the data
    '''
    std = np.std(data)

    return [std]


def skewness(data):
    '''
    Returns the skewness of the data
    '''
    skewness = scipy.stats.skew(data)

    return [skewness]


def kurtosis(data):
    '''
    Returns kurtosis calculated from the data
    '''
    kurtosis = scipy.stats.kurtosis(data)

    return [kurtosis]


def sum_of_positive_derivative(data):
    '''
    Retruns the sum of positive values of the first derivative of the data
    '''
    first_derivative = np.diff(data, n=1)
    pos_sum = sum(d for d in first_derivative if d > 0)

    return [pos_sum]


def sum_of_negative_derivative(data):
    '''
    Returns the sum of the negative values of the first derivative of the data
    '''
    first_derivative = np.diff(data, n=1)
    neg_sum = sum(d for d in first_derivative if d < 0)

    return [neg_sum]


def mean(data):
    '''
    Returns the mean of the data
    '''
    mean = np.mean(data)

    return [mean]


def median(data):
    '''
    Returns the median of the data
    '''
    median = np.median(data)

    return [median]


def range(data):
    '''
    Retruns the range of the data
    '''
    range = max(data) - min(data)

    return [range]


def maximum(data):
    '''
    Returns the maximum of the data
    '''
    return [max(data)]


def minimum(data):
    '''
    Returns the minimum of the data
    '''
    return [min(data)]

def get_frequencies(data):
    freq_data = np.abs(np.fft.fftshift(data))
    features = (np.mean(freq_data) + np.median(freq_data) +
                np.std(freq_data) +
                np.max(freq_data) + np.min(freq_data) + (np.max(freq_data)-np.min(freq_data)))

    return features


