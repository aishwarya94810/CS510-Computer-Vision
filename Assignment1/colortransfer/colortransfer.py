import numpy 
import cv2

# this exercise references "Color Transfer between Images" by Reinhard et al.

numpyFrom = cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment1/colortransfer/transfer-from.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

numpyTo = cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment1/colortransfer/transfer-to.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0


# in order for make matching the statistics more meaningful, the images are first converted to the LAB color space

numpyFrom = cv2.cvtColor(src=numpyFrom, code=cv2.COLOR_BGR2Lab)
numpyTo = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_BGR2Lab)

# match the color statistics of numpyTo to those of numpyFrom

# a)calculate the per-channel mean of the data points / pixels of numpyTo, and subtract these from numpyTo according to equation 10
# b)calculate the per-channel std of the data points / pixels of numpyTo and numpyFrom, and scale numpyTo according to equation 11
# c)calculate the per-channel mean of the data points / pixels of numpyFrom, and add these to numpyTo according to the description after equation 11



xnumpyTo = numpyTo[:,:,0]
ynumpyTo = numpyTo[:,:,1]
znumpyTo = numpyTo[:,:,2]

xnumpyFrom = numpyFrom[:,:,0]
ynumpyFrom = numpyFrom[:,:,1]
znumpyFrom = numpyFrom[:,:,2]

# a)calculate the per-channel mean of the data points / pixels of numpyTo,
xnumpyTomean = numpy.mean(xnumpyTo)

ynumpyTomean = numpy.mean(ynumpyTo)

znumpyTomean = numpy.mean(znumpyTo)

# subtract these from numpyTo according to equation 10
x1 = xnumpyTo - xnumpyTomean

y1 = ynumpyTo - ynumpyTomean

z1 = znumpyTo - znumpyTomean

# b)calculate the per-channel std of the data points / pixels of numpyTo 
xnumpyTostd = numpy.std(x1)

ynumpyTostd = numpy.std(y1)

znumpyTostd = numpy.std(z1)

# calculate the per-channel std of the data points / pixels of numpyFrom
xnumpyFromstd = numpy.std(x1)

ynumpyFromstd = numpy.std(y1)

znumpyFromstd = numpy.std(z1)

# scale numpyTo according to equation 11

x2 = (xnumpyFromstd/xnumpyTostd) * x1

y2 = (ynumpyFromstd/ynumpyTostd) * y1
z2 = (znumpyFromstd/znumpyTostd) * z1

# c)calculate the per-channel mean of the data points / pixels of numpyFrom,
xnumpyFromMean = numpy.mean(xnumpyFrom)
ynumpyFromMean = numpy.mean(ynumpyFrom)
znumpyFromMean = numpy.mean(znumpyFrom)


# add these to numpyTo according to the description after equation 11
xfinal = xnumpyFromMean + x2
yfinal = ynumpyFromMean + y2
zfinal = znumpyFromMean + z2

numpyTo = numpy.stack((xfinal,yfinal,zfinal),-1)

# after matching the statistics, some of the intensity values might be out of the valid range and are hence clipped / clamped

numpyTo[:, :, 0] = numpyTo[:, :, 0].clip(0.0, 100.0)
numpyTo[:, :, 1] = numpyTo[:, :, 1].clip(-127.0, 127.0)
numpyTo[:, :, 2] = numpyTo[:, :, 2].clip(-127.0, 127.0)

# finaly, the matched image is being converted back to the RGB color space

numpyOutput = cv2.cvtColor(src=numpyTo, code=cv2.COLOR_Lab2BGR)

cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment1/02-colortransfer.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
