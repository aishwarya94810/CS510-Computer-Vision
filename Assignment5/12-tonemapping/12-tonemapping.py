import numpy
import cv2
import os
import zipfile
import matplotlib.pyplot

# this exercise references "Photographic Tone Reproduction for Digital Images" by Reinhard et al.

numpyRadiance = cv2.imread(filename='./samples/ahwahnee.hdr', flags=-1)

# perform tone mapping according to the photographic luminance mapping

# first extracting the intensity from the color channels
# note that the eps is to avoid divisions by zero and log of zero

x,y,z = numpyRadiance.shape
N=x*y

numpyIntensity = cv2.cvtColor(src=numpyRadiance, code=cv2.COLOR_BGR2GRAY) + 0.0000001

# start off by approximating the key of numpyIntensity according to equation 1

print(N)

Lw= numpy.exp((1/N)*(numpy.sum(numpy.log(numpyIntensity))))


# then normalize numpyIntensity using a = 0.18 according to equation 2
Lxy= (0.18/Lw)*(numpyIntensity)

# afterwards, apply the non-linear tone mapping prescribed by equation 3

Ldxy= Lxy/(1+Lxy)

# finally obtain numpyOutput using the ad-hoc formula with s = 0.6 from the slides

numpyOutput = numpyRadiance / numpyIntensity[:, :, None]
numpyOutput = numpy.power(numpyOutput, 0.6)
numpyOutput = numpyOutput * Ldxy[:, :, None]

cv2.imwrite(filename='./samples/12-tonemapping.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))