import numpy as np
import scipy
import math
from scipy.integrate import simps

dx = 7

def get_feature_headers_for_ps_spectrum():
    return ['PS - AUC 400Hz-100Hz', 'PS - db increase 400Hz to 600Hz', 'PS - db decrease 400Hz to 600Hz', 'PS - dB change 100Hz to 200Hz', 'PS - db increase 200Hz to 400Hz']

def get_features_for_ps_spectrum(spectrum):
    return np.array([
        get_area_under_curve(spectrum, 400, 1000),
        get_power_change(spectrum, 400, 600),
        get_power_change(spectrum, 100, 200),
        get_power_change(spectrum, 200, 400),
    ])

def get_feature_headers_for_bs_spectrum():
    return ['BS - AUC 100 to 2500Hz', 'BS - Peak 300 to 500Hz']

def get_features_for_bs_spectrum(spectrum):
    return np.array([
        get_area_under_curve(spectrum, 100, 2500),
        peak_freq(spectrum, 300, 500),
    ])

def get_feature_headers_for_tf_spectrum():
    return ['TF - AUC 100 to 200Hz']

def get_features_for_tf_spectrum(spectrum):
    return np.array([
        get_area_under_curve(spectrum, 100, 200),
    ])

def get_area_under_curve(spectrum, lower_freq, higher_freq):
    # https://stackoverflow.com/questions/13320262/calculating-the-area-under-a-curve-given-a-set-of-coordinates-without-knowing-t
    lower_index_inc = get_lower_bound_index(lower_freq) - 1
    higher_index_exc = get_higher_bound_index(higher_freq)
    section = spectrum[lower_index_inc:higher_index_exc]
    powers = [row[1] for row in section]
    return scipy.integrate.trapz(powers, dx=dx)

def get_power_change(spectrum, from_freq, to_freq):
    from_index = closest_index(from_freq)
    to_index = closest_index(to_freq)
    return spectrum[from_index][1] - spectrum[to_index][1]

def peak_freq(spectrum, lower_freq, higher_freq):
    lower_index_inc = get_lower_bound_index(lower_freq) - 1
    higher_index_exc = get_higher_bound_index(higher_freq)
    section = spectrum[lower_index_inc:higher_index_exc, :]
    _, peak_freq = section.max(axis=0)
    return peak_freq

def get_lower_bound_index(freq):
    index = math.ceil(freq / dx)
    if (freq % dx != 0):
        index = index + 1
    return int(index)

def get_higher_bound_index(freq):
    index = math.ceil(freq / dx)
    if (freq % dx != 0):
        index = index + 1
    return int(index + 1)

def closest_index(freq):
    return int(round(freq / dx) + 1)
