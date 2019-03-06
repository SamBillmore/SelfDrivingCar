import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
	'''
	Converts a colour image to a gradient image using the canny algorithm
	'''
	# Convert to grayscale (so we only have a single channel (intensity value) to process)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# Apply Gaussian blur to reduce noise from image (NB: canny will do this by default)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	# Apply canny algorithm to find high changes in gradient
	canny = cv2.Canny(blur, 50, 150)
	return canny

def region_of_interest(image):
	'''
	Creates a triangular region of interest with zero values outside the triangle
	'''
	height = image.shape[0]
	# Define triangle vertices
	polygons = np.array([
		[(200,height),(1100,height),(550,250)]
		])
	# Create np array of same size as image with zeros in all positions 
	mask = np.zeros_like(image)
	# Set area within triangle to be white
	cv2.fillPoly(mask,polygons,255)
	# Apply mask to image using bitwise '&'
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image




image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)

# Show image
cv2.imshow('result', cropped_image)
cv2.waitKey(0)
