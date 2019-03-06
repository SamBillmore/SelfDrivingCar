import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_gradient_image(image):
	'''
	Converts a colour image to a gradient image using the canny algorithm
	'''
	# Convert to grayscale (so we only have a single channel (intensity value) to process)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# Apply Gaussian blur to reduce noise from image (NB: canny will do this by default)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	# Apply canny algorithm to find high changes in gradient
	gradient_image = cv2.Canny(blur, 50, 150)
	return gradient_image

def region_of_interest(image):
	'''
	Isolates a triangular region of interest with zero values outside the triangle and the image inside
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

def display_lines(image,lines):
	'''
	Plots lines on black background the size of image 
	'''
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
	return line_image



image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
gradient_image = create_gradient_image(lane_image)
cropped_image = region_of_interest(gradient_image)
# Identify lines using Hough space
lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# Show original image
cv2.imshow('result', lane_image)
cv2.waitKey(0)

# Show image with identified lane markers
cv2.imshow('result', combo_image)
cv2.waitKey(0)
