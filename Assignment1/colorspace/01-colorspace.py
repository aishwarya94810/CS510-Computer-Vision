import numpy 
import cv2

# this exercise references "Color Transfer between Images" by Reinhard et al.

numpyInput = cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment1/colorspace/fruits.png', flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0

#mulitplying each component with r, g and b components to get L, M and S values.



# the two ways i see for doing this (there are others as well though) are as follows
# either iterate over each pixel, performing the matrix-vector multiplication one by one and storing the result in a pre-allocated numpyOutput
# or split numpyInput into its three channels, linearly combining them to obtain the three converted color channels, before using numpy.stack to merge them

# keep in mind that that opencv arranges the color channels typically in the order of blue, green, red

# convert numpyInput to the LMS color space and store it in numpyOutput according to equation 4

"""
r=numpyInput[:,:,0]
g=numpyInput[:,:,1]
b=numpyInput[:,:,2]
"""


L = float(0.3811) * numpyInput[0:,0:,2] + float(0.5783) * numpyInput[0:,0:,1] + float(0.0402) * numpyInput[0:,0:,0]
M = float(0.1967) * numpyInput[0:,0:,2] + float(0.7244) * numpyInput[0:,0:,1] + float(0.0782) * numpyInput[0:,0:,0]
S = float(0.0241) * numpyInput[0:,0:,2] + float(0.1288) * numpyInput[0:,0:,1] + float(0.8444) * numpyInput[0:,0:,0]

numpyOutput = numpy.stack((L,M,S),-1) #combining the L,M and S elements


cv2.imwrite('/Macintosh HD⁩/⁨Users⁩/⁨aishu⁩/⁨Documents⁩/⁨GitHub⁩/⁨CS510-Computer-Vision-⁩/⁨Assignment1⁩/⁨colorspace⁩/01-colorspace.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

