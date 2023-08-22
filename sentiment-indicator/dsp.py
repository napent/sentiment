from scipy.signal import butter, lfilter, lfilter_zi


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    zi = lfilter_zi(b, a) * data[0]

    y, _ = lfilter(b, a, data, zi=zi)
    return y


def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    zi = lfilter_zi(b, a) * data[0]

    y, _ = lfilter(b, a, data, zi=zi)
    return y


def process_low_pass(df, cutoff):
    order = 1
    fs = 10
    return butter_lowpass_filter(df, cutoff, fs, order)


def process_high_pass(df, cutoff):
    order = 1
    fs = 10
    return butter_highpass_filter(df, cutoff, fs, order)
