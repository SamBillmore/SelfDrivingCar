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

def region_of_interest(image,v1:(float,float),v2:(float,float),v3:(float,float)):
    '''
    Isolates a triangular region of interest with zero values outside the triangle and the image inside
    
    Vertices of triangle are defined by tuples by reference to the proportion of:
    - width (measured from the left of the image) and 
    - height (measured from the top of the image)
    
    Examples:
    v1 = (1,1) => this vertex would be in the bottom right hand corner of the image
    v2 = (0.5,0.5) => this vertex would be in the centre of the image
    v3 = (0.25,0.75) => this vertex would be 1/4 of the way from the left of the image and
        3/4 of the way from the top of the image
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Define triangle using vertices
    polygons = np.array([
        [(int(v1[0]*width),int(v1[1]*height)),(int(v2[0]*width),int(v2[1]*height)),(int(v3[0]*width),int(v3[1]*height))]
    ])
    # Create np array of same size as image with zeros in all positions 
    mask = np.zeros_like(image)
    # Set area within triangle to be white
    cv2.fillPoly(mask,polygons,255)
    # Apply mask to image using bitwise '&'
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def make_line(image,line_parameters,proportion):
	'''
	Identifies end points for a line starting at the bottom of the image with:
	- slope and intercept defined by line_parameters
	- length of line defined by proportion, being the proportion of the height measured from the top of the image the line should reach
	'''
	slope, intercept = line_parameters
	height = image.shape[0]
	y1 = height
	y2 = int(y1*proportion)
	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)
	return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
	'''
	Creates end points for two lines, one from the average lines with positive slope and one from the average lines with negative slope

	If there are no lines identified with positive slope or no lines with negative slope then the default position is 0,0
	'''
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2),(y1,y2),1)
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))
	if len(left_fit) > 0:
		left_fit_average = np.average(left_fit, axis=0)
		left_line = make_line(image,left_fit_average,0.6)
	else:
		left_line = [0,0,0,0]
	if len(right_fit) > 0:
		right_fit_average = np.average(right_fit, axis=0)
		right_line = make_line(image,right_fit_average,0.6)
	else:
		right_line = [0,0,0,0]
	return np.array([left_line,right_line])

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


def lane_finder(image):
	'''
	Takes an image and identifies the straight lines within a defined area of interest
	Returns images from each stage of the algorithm
	'''
	gradient_image = create_gradient_image(image)
	cropped_image = region_of_interest(gradient_image,v1=(0.15,1),v2=(0.86,1),v3=(0.43,0.36))
	# Identify lines using Hough space
	lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
	line_image = display_lines(image,lines)
	averaged_lines = average_slope_intercept(image,lines)
	average_line_image = display_lines(image,averaged_lines)
	combo_image = cv2.addWeighted(image, 0.8, average_line_image, 1, 1)
	return gradient_image, cropped_image, line_image, average_line_image, combo_image

def show_steps_for_image(image):
	'''
	'''
	gradient_image, cropped_image, line_image, average_line_image, combo_image = lane_finder(image)
	# Show original image
	cv2.imshow('result', image)
	cv2.waitKey(0)

	# Show gradient image
	cv2.imshow('result', gradient_image)
	cv2.waitKey(0)

	# Show cropped gradient image
	cv2.imshow('result', cropped_image)
	cv2.waitKey(0)

	# Show lines identified
	cv2.imshow('result', line_image)
	cv2.waitKey(0)

	# Show averaged lines identified
	cv2.imshow('result', average_line_image)
	cv2.waitKey(0)

	# Show original image with identified lane markers
	cv2.imshow('result', combo_image)
	cv2.waitKey(0)


# Apply lane finding algorithm to a single image
source_image = cv2.imread('./Images_and_videos/test_image.jpg')
lane_image = np.copy(source_image)
show_steps_for_image(lane_image)

# Apply lane finding algorithm to a video
video = cv2.VideoCapture('./Images_and_videos/test2.mp4')
while(video.isOpened()):
	_, frame = video.read()
	_, _, _, _, frame_combo_image = lane_finder(frame)
	cv2.imshow('result', frame_combo_image)
	if cv2.waitKey(1) == ord('q'):
		break
video.release()
cv2.destroyAllWindows()
