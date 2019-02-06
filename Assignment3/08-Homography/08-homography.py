import numpy
import cv2


numpyInput = cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment3/Homography/homography-2.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# create the mapping between the four corresponding points

intSrc = [ [266, 343], [646, 229], [388, 544], [777, 538] ]
intDst = [ [302, 222], [746, 231], [296, 490], [754, 485] ]

# construct the linear homogeneous system of equations
# use a singular value decomposition to solve the system
# in practice, cv2.findHomography can be used for this
# however, do not use this function for this exercise

Alist = []

for lhomog in range(len(intSrc)):
	sx, sy = intSrc[lhomog][0], intSrc[lhomog][1]
	dx, dy = intDst[lhomog][0], intDst[lhomog][1]

	Alist.append([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx, -dx])
	Alist.append([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy, -dy])

# h1 = numpy.array([h11,h12,h13,h21,h22,h23,h31,h32])

# h2 = h1.reshape(8,1)

X, Y, Z = numpy.linalg.svd(numpy.array(Alist, numpy.float32))


numpyHomography = Z[-1, :].reshape(3, 3) / Z[-1, -1]

# use a backward warping algorithm to warp the source
# to do so, we first create the inverse transform
# use bilinear interpolation for resampling
# in practice, cv2.warpPerspective can be used for this
# however, do not use this function for this exercise

numpyHomography = numpy.linalg.inv(numpyHomography)

numpyOutput = numpy.zeros(numpyInput.shape, numpy.float32)

for CorY in range(numpyInput.shape[0]):
	for CorX in range(numpyInput.shape[1]):
		numpyDest = numpy.array([ CorX, CorY, 1.0 ], numpy.float32)

		numpySource = numpy.matmul(numpyHomography, numpyDest.T)
		numpySource = numpySource / numpySource[2]

		if numpySource[0] < 0.0 or numpySource[0] > numpyOutput.shape[1] - 1.0:
			continue

		elif numpySource[1] < 0.0 or numpySource[1] > numpyOutput.shape[0] - 1.0:
			continue

		# end

		a = numpySource[0] - int(numpy.floor(numpySource[0])) 
		b =  numpySource[1] - int(numpy.floor(numpySource[1]))

		numpyOutput[CorY, CorX] = (1 - a) * (1 - b) * numpyInput[int(numpySource[1]), int(numpySource[0])] + (1 - a) * b * numpyInput[int(numpySource[1]) + 1, int(numpySource[0])] + a * (1 - b) * numpyInput[int(numpySource[1]), int(numpySource[0]) + 1] + a * b * numpyInput[int(numpySource[1]) + 1, int(numpySource[0]) + 1]

		cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment3/Homography/08-homography.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))


	# end
# end