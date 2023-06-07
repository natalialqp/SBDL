from scipy import signal

class Filters(object):
    def median_filter(self, data):
        '''
        This function performs the filtering of the signal according to the median filter
        data -> vector
        '''
        y_mean = signal.medfilt(data)
        return y_mean