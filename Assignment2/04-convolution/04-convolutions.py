import numpy
import cv2

# this exercise references "Interactions Between Color Plane Interpolation and Other Image Processing Functions in Electronic Photography" by Adams

numpyInput = cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment2/04-convolution/demosaicing.tif', flags=cv2.IMREAD_GRAYSCALE).astype(numpy.float32) / 255.0

numpyOutput = numpy.zeros([numpyInput.shape[0], numpyInput.shape[1], 3], numpy.float32)

# demosaic numpyInput by using convololutions to mimic bilinear interpolation as shown in the slides and described in section 3.3

# the input has the following beyer pattern, id est that the top left corner is red

# RGRGRG ....
# GBGBGB ....
# RGRGRG ....
# GBGBGB ....
# ...........
# ...........

# to do this using convolutions, the first step is to separate the colors from numpyInput into four channels such that numpyOutput[:, :, 1] for example becomes

# 0G0G0G ....
# G0G0G0 ....
# 0G0G0G ....
# G0G0G0 ....
# ...........
# ...........

# since this can be tricky and you might not be perfectly familiar with indexing and splicing matrices yet, this is already done for you below

#G component
matrixG =numpy.array([[0.00,0.25,0.00],
		  			 [0.25,1.00,0.25],
	  	  			 [0.00,0.25,0.00]])

#R component
matrixR =numpy.array([[0.25,0.50,0.25],
	  	 			 [0.50,1.00,0.50],
	  	  			 [0.25,0.50,0.25]])

#B component
matrixB =numpy.array([[0.25, 0.50 , 0.25],
	  	  			 [0.50, 1.00  , 0.50],
	      			 [0.25, 0.50 , 0.25]])

numpyOutput[1::2, 1::2, 0] = numpyInput[1::2, 1::2]
numpyOutput[0::2, 1::2, 1] = numpyInput[0::2, 1::2]
numpyOutput[1::2, 0::2, 1] = numpyInput[1::2, 0::2]
numpyOutput[0::2, 0::2, 2] = numpyInput[0::2, 0::2]



numpyOutput[:, :, 0]= cv2.filter2D (numpyOutput[:, :, 0],-1,matrixR,cv2.BORDER_DEFAULT)
numpyOutput[:, :, 1]= cv2.filter2D (numpyOutput[:, :, 1],-1,matrixG,cv2.BORDER_DEFAULT)
numpyOutput[:, :, 2]= cv2.filter2D (numpyOutput[:, :, 2],-1,matrixB,cv2.BORDER_DEFAULT)


# for each channel in numpyOutput, use a suitable convolution to perform the demosaicing and store the output back in its respective channel in numpyOutput
# we already discussed in class what the appropriate kernel for the green channel is, determining the kernel for the other two channels is up to you
# to be able to convolve a channel from numpyOutput and storing the result back in the same channel, the convolution must not affect the resolution
# we need padding for the convolution to not affect the resolution, this is already built into OpenCV but make sure to use cv2.BORDER_DEFAULT as the border type

cv2.imwrite(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment2/04-convolution/04-convolution.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))
