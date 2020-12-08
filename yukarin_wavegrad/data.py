import numpy


def encode_mulaw(x: numpy.ndarray, mu: int = 256):
    mu = mu - 1
    y = numpy.sign(x) * numpy.log1p(mu * numpy.abs(x)) / numpy.log1p(mu)
    return y


def decode_mulaw(x: numpy.ndarray, mu: int = 256):
    mu = mu - 1
    y = numpy.sign(x) * ((1 + mu) ** numpy.abs(x) - 1) / mu
    return y
