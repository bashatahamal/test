import numpy as np
import argparse
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())


def NormalizeData(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))


def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]

			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
	return boxes[pick]


def MatchTemplate(template, image, threshold, nms_thresh):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	boundingBoxes=np.ones((1,4), dtype=int)
	# loop over the scales of the image (start stop numstep) from the back
	for scale in np.linspace(0.2, 2.0, 40)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])

		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
		
		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		#edged = cv2.Canny(resized, 50, 200)

		#Otsu threshold
		#ret_img,resized = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		#Simple threshold
		#ret_img, resized = cv2.threshold(resized,127,255,cv2.THRESH_BINARY)

		#Adaptive threshold value is the mean of neighbourhood area
		#resized = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

		#Adaptive threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
		resized = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


		#cv2.imshow("edged",resized)
		result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
		#result = NormalizeData(result)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		#print(maxVal)
		# check to see if the iteration should be visualized
		if args.get("visualize", False):
			# draw a bounding box around the detected region
			#clone = np.dstack([edged, edged, edged])
			clone = np.dstack([resized, resized, resized])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)

		#threshold = 0.7
		# loc = np.where(result >= threshold)
		# (startX, startY) = (int(loc[0] * r), int(loc[1] * r))
		# (endX, endY) = (int((loc[0] + tW) * r), int((loc[1] + tH) * r))
		# for pt in zip(*loc[::-1]):
		# 	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		# 	#cv2.rectangle(image, pt, (pt[0] + tW , pt[1] + tH), (0, 0, 255), 2)
		
		temp=[]
		if maxVal > threshold:
			found = (maxVal, maxLoc, r)
			(_, maxLoc, r) = found
			(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
			(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
			# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			temp = [startX, startY, endX, endY]
			boundingBoxes = np.append(boundingBoxes, [temp], axis=0)
			
			print(maxVal)

	# unpack the bookkeeping varaible and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	# (_, maxLoc, r) = found
	# (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	# (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	
	#if detected 
	if len(boundingBoxes) > 1 :
		boundingBoxes = np.delete(boundingBoxes, 0, axis = 0)
		pick = non_max_suppression_slow(boundingBoxes, nms_thresh)
		print (len(pick))
		print (pick)
		# print (boundingBoxes)
		for (startX, startY, endX, endY) in pick:
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	return image


# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#template = cv2.Canny(template, 50, 200)

#Otsu threshold
#ret_temp,template = cv2.threshold(template,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Simple threshold
#ret_temp, template = cv2.threshold(template,127,255,cv2.THRESH_BINARY)

#Adaptive threshold value is the mean of neighbourhood area
#template = cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

#Adaptive threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
template = cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

cv2.imshow("template", template)
(tH, tW) = template.shape[:2]
#cv2.imshow("Template", template)


# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.png"):
	
	image = cv2.imread(imagePath)
	threshold = 0.7
	nms_thresh = 0.3
	result = MatchTemplate (template, image, threshold, nms_thresh)
	cv2.imshow("Match", result)
	cv2.waitKey(0)

cv2.destroyAllWindows()
