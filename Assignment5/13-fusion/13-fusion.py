import numpy
import cv2

# this exercise references "Exposure Fusion" by Mertens et al.

numpyInputs = [
	cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment5/fusion-1.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0,
	cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment5/fusion-2.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0,
	cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment5/fusion-3.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
]

# use the quality measures to extract a weight map for each image according to section 3.1
numpyWeights = []
numpyWeights_sum = numpy.zeros(numpyInputs[0].shape[:2], dtype=numpy.float32)
# set the weighting exponents to one, thus equaling the contrition of contrast, saturation, and exposedness
(w_c, w_s, w_e) = (1, 1, 1)
# make sure to add a small epsilon to each weight map to avoid divisions by zero in the subsequent step
	

i = 0

for image in numpyInputs:
	# print(image.shape[:])
	W = numpy.ones(image.shape[:2], dtype=numpy.float32)

	# contrast
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	laplacian = cv2.Laplacian(image_gray, cv2.CV_32F)
	W_contrast = numpy.absolute(laplacian) ** w_c
	W = numpy.multiply(W, W_contrast)

	# saturation
	W_saturation = image.std(axis=2,dtype=numpy.float32) ** w_s
	W = numpy.multiply(W, W_saturation)

	# well-exposedness
	sigma2 = 0.2
	W_exposedness = numpy.prod(numpy.exp(-((image - 0.5)**2)/(2*(sigma2**2))),axis=2, dtype=numpy.float32) ** w_e
	W = numpy.multiply(W, W_exposedness)
	
	W += 0.0000001
	numpyWeights_sum += W 
	numpyWeights.append(W)
# normalize the weight maps such that they sum up to one at each pixel as described in section 3.2
# store the three weight maps in the numpyWeights array which will be used below to perform the blending

# normalization
nonzero = numpyWeights_sum > 0
for i in range(len(numpyWeights)):
	numpyWeights[i][nonzero] /= numpyWeights_sum[nonzero]

# creating the laplacian and gaussian pyramids to perform multiband blending
# defining separate functions for this steps makes the code easier to read

def gaussian_pyramid(numpyInput, intLevels):
	numpyPyramid = [ numpyInput ]

	for intLevel in range(intLevels):
		numpyPyramid.append(cv2.pyrDown(numpyPyramid[-1]))
	# end

	return numpyPyramid
# end

def laplacian_pyramid(numpyInput, intLevels):
	numpyPyramid = [ numpyInput ]

	for intLevel in range(intLevels):
		numpyPyramid.append(cv2.pyrDown(numpyPyramid[-1]))

		numpyPyramid[-2] -= cv2.pyrUp(numpyPyramid[-1])
	# end

	return numpyPyramid
# end

numpyInputs = [ laplacian_pyramid(numpyInput, 6) for numpyInput in numpyInputs ]
numpyWeights = [ gaussian_pyramid(numpyWeight, 6) for numpyWeight in numpyWeights ]

# constructing a laplacian pyramid by using the weights from the gaussian pyramid
# eventually obtaining the fused result by recovering the output from the merged pyramid

numpyPyramid = []

for intLevel in range(len(numpyInputs[0])):
	numpyPyramid.append(sum([ numpyInputs[intInput][intLevel] * numpyWeights[intInput][intLevel][:, :, None] for intInput in range(len(numpyInputs)) ]))
# end

numpyOutput = numpyPyramid.pop(-1)

while len(numpyPyramid) > 0:
	numpyOutput = cv2.pyrUp(numpyOutput) + numpyPyramid.pop(-1)
# end

cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment5/13-fusion-1.png', img=(numpyWeights[0][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment5/13-fusion-2.png', img=(numpyWeights[1][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment5/13-fusion-3.png', img=(numpyWeights[2][0] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment5/13-fusion-4.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))