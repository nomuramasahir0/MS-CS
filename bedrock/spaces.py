import numpy as np


class Space(object):
    def __init__(self, continuous_space):
        self._continuous_space = continuous_space

    @property
    def continuous_space(self):
        return self._continuous_space


class SearchRange(object):
    def __init__(self, label):
        self.label = label

    def convert(self, x):
        raise NotImplementedError


class ContinuousSearchRange(SearchRange):
    def __init__(self, label, low, high, clipping=True):
        super().__init__(label)
        self._low = low
        self._high = high
        self._clipping = clipping

    def convert(self, x):
        raise NotImplementedError

    def inverse_convert(self, converted_x):
        raise NotImplementedError


class Uniform(ContinuousSearchRange):
    def __init__(self, label, low, high, clipping=True):
        super().__init__(label, low, high, clipping)

    def convert(self, x):
        converted_x = self._low + (self._high - self._low) * x
        return np.clip(converted_x, self._low,
                       self._high) if self._clipping else converted_x

    def inverse_convert(self, converted_x):
        x = (converted_x - self._low) / (self._high - self._low)
        assert 0. <= x <= 1.
        return x


class QUniform(ContinuousSearchRange):
    def __init__(self, label, low, high, clipping=True):
        super().__init__(label, low, high, clipping)

    def convert(self, x):
        converted_x = np.floor(self._low + (self._high - self._low + 1) * x)
        return int(np.clip(converted_x, self._low, self._high) if self._clipping else converted_x)

    def inverse_convert(self, converted_x):
        x = (converted_x - self._low) / (self._high - self._low)
        assert 0. <= x <= 1.
        return x


class PowUniform(ContinuousSearchRange):
    def __init__(self, label, low, high, b, clipping=True):
        super().__init__(label, np.log10(low), np.log10(high), clipping)
        self._b = b

    def convert(self, x):
        converted_x = self._low + (self._high - self._low) * x
        converted_x = np.clip(converted_x, self._low,
                              self._high) if self._clipping else converted_x
        return np.power(self._b, converted_x)

    def inverse_convert(self, converted_x):
        x = np.log10(converted_x / self._low) / (np.log10(self._high) - np.log10(self._low))
        assert 0. <= x <= 1.
        return x


class PowQUniform(ContinuousSearchRange):
    def __init__(self, label, low, high, b, clipping=True):
        super().__init__(label, np.log10(low), np.log10(high), clipping)
        self._b = b

    def convert(self, x):
        converted_x = self._low + (self._high - self._low) * x
        converted_x = np.clip(converted_x, self._low,
                              self._high) if self._clipping else converted_x
        return int(np.power(self._b, converted_x))

    def inverse_convert(self, converted_x):
        x = np.log10(converted_x / self._low) / (np.log10(self._high) - np.log10(self._low))
        assert 0. <= x <= 1.
        return x
