import numpy
import cv2

# this exercise references "The Laplacian Pyramid as a Compact Image Code" by Burt and Adelson

numpyInput = cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment3/pyramid/lenna.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

# create a laplacian pyramid with four levels as described in the slides as well as in the referenced paper

# the following iterates over the levels in numpyPyramid and saves them as an image accordingly
# level four is just a small-scale representation of the original input image and can be saved as usual
# the value range for the other levels are outside of [0, 1] and a color mapping is applied before saving them

numpyPyramid = []


for i in range(4):
	if i ==3:
		numpyPyramid.append(numpyInput)
		break
	pdown =cv2.pyrDown(numpyInput)
	pup= cv2.pyrUp(pdown)
	#Laplace= cv2.subtract(numpyInput- pup)
	numpyPyramid.append(numpyInput- pup)
	numpyInput=pdown
		


for intLevel in range(len(numpyPyramid)):
	if intLevel == len(numpyPyramid) - 1:
		#down = cv2.pyrDown(numpyInput)
		cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment3/pyramid/07-pyramid	.png' + str(intLevel + 1) + '.png', img=(numpyPyramid[intLevel] * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

	elif intLevel != len(numpyPyramid) - 1:
		
		cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment3/pyramid/07-pyramid.png' + str(intLevel + 1) + '.png', img=cv2.applyColorMap(src=((numpyPyramid[intLevel] + 0.5) * 255.0).clip(0.0, 255.0).astype(numpy.uint8), colormap=cv2.COLORMAP_COOL))
		#numpyInput=down
	# end
# end