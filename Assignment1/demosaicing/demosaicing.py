import numpy
import cv2
# this exercise references "Interactions Between Color Plane Interpolation and Other Image Processing Functions in Electronic Photography" by Adams


numpyInput = cv2.imread(filename='/home/aishwarya/Desktop/Term2/IntroVisual/Assignment1/demosaicing/demosaicing.tif', flags=cv2.IMREAD_GRAYSCALE).astype(numpy.float32) / 255.0

numpyOutput = numpy.zeros([numpyInput.shape[0]-2, numpyInput.shape[1]-2, 3], numpy.float32)

# demosaic numpyInput by using bilinear interpolation as shown in the slides and described in section 3.3

# the input has the following beyer pattern, id est that the top left corner is red

# RGRGRG ....
# GBGBGB ....
# RGRGRG ....
# GBGBGB ....
# ...........
# ...........

# the straightforward way that i see for doing this (there are others as well though) is to iterate over each pixel and resolving each of the four possible cases
# to simplify this, you can iterate from (1 to numpyInput.shape[0] - 1) and (1 to numpyInput.shape[1] - 1) to avoid corner cases, numpyOutput is accordingly one pixel smaller on each side

# notice that to fill in the missing greens, you will always be able to take the average of four neighboring values
# however, depending on the case, you either get four or only two neighboring values for red and blue
# this is perfectly fine, in this case you can simply use the average of two values if only two neighbors are available

for i in range(1, rows):
	for j in range(1, columns):
		if i % 2 == 1: #rows: 1,3,5,7..
			if j % 2 == 1: #columns: 1,3,5,7...
				numpyOutput[i,j,0] = numpyInput[i,j]
				if i == 1: #handling all two B values at R pixel and G is always 4. 
					numpyOutput[i,j,2] = (numpyInput[i-1, j-1] + numpyInput[i+1, j-1] + numpyInput[i-1, j+1] + numpyInput[i+1, j+1]) / 4
					numpyOutput[i,j,1] = (numpyInput[i, j-1] + numpyInput[i, j+1] + numpyInput[i-1, j] + numpyInput[i+1, j]) / 4
				if j == 1:
					numpyOutput[i,j,1] = (numpyInput[i, j-1] + numpyInput[i, j+1] + numpyInput[i-1, j] + numpyInput[i+1, j]) / 4
					numpyOutput[i,j,2] = (numpyInput[i-1, j-1] + numpyInput[i+1, j-1] + numpyInput[i-1, j+1] + numpyInput[i+1, j+1]) / 4

				elif i > 1 and j > 1:
					numpyOutput[i,j,2] = (numpyInput[i-1, j-1] + numpyInput[i+1, j-1] + numpyInput[i-1, j+1] + numpyInput[i+1, j+1]) / 4
					numpyOutput[i,j,1] = (numpyInput[i, j-1] + numpyInput[i, j+1] + numpyInput[i-1, j] + numpyInput[i+1, j]) / 4
			else:
				numpyOutput[i,j,1] = numpyInput[i,j]
				numpyOutput[i,j,2] = (numpyInput[i+1, j] + numpyInput[i-1, j]) / 2#red top bottom
				numpyOutput[i,j,0] = (numpyInput[i,j-1] + numpyInput[i, j+1]) / 2# b sides
		else: #2,4,6,8
			if j % 2 == 1:
				numpyOutput[i,j,1] = numpyInput[i,j]
				numpyOutput[i,j,2] = (numpyInput[i,j-1] + numpyInput[i, j+1]) / 2#red sides 2
				numpyOutput[i,j,0] = (numpyInput[i+1, j] + numpyInput[i-1, j]) / 2#b top bottom 2 
			else:
				numpyOutput[i,j,2] = numpyInput[i,j]
				numpyOutput[i,j,1] = (numpyInput[i, j-1] + numpyInput[i, j+1] + numpyInput[i-1, j] + numpyInput[i+1, j]) / 4
				numpyOutput[i,j,0] = (numpyInput[i+1, j] + numpyInput[i-1, j]) / 2#b top bottom 2
				
	
cv2.imwrite(filename='./03-demosaicing.png', img=(numpyOutput * 255.0).clip(0.0, 255.0).astype(numpy.uint8))

