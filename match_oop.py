import numpy as np 
import argparse
import imutils
import glob
import cv2

#__argument parser 
ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True, help="Path to matched images")
ap.add_argument("-v", "--visualize", help="Flag indicating wether or not to visualize each iteration")
args = vars(ap.parse_args())

class Marker:
    def __init__ (self, **kwargs):
        self._Data=kwargs
    
    def GetTemplate_Location(self):
        return self._Data["template_loc"]
    def GetImage_Location(self):
        return self._Data["image_loc"]
    def GetTemplate_Threshold(self):
        return self._Data["template_thresh"]
    def GetNMS_Threshold(self):
        return self._Data["nms_thresh"]

    def non_max_suppression_slow(self, boxes, overlapThresh):
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
    
    def Match_Template(self, visualize=False):
        #__get template
        template = cv2.imread(self.GetTemplate_Location())
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        ##Canny edge
        #template = cv2.Canny(template, 50, 200)
        ##Otsu threshold
        #ret_temp,template = cv2.threshold(template,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ##Simple threshold
        #ret_temp, template = cv2.threshold(template,127,255,cv2.THRESH_BINARY)
        ##Adaptive threshold value is the mean of neighbourhood area
        #template = cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        ##Adaptive threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
        template = cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        cv2.imshow("Template", template)
        (tH,tW) = template.shape[:2]

        #__get image
        image = cv2.imread(self.GetImage_Location())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boundingBoxes = np.ones((1,4), dtype=int)
        max_value_list = []
        #__loop over scaled image (start stop numstep) from the back
        for scale in np.linspace(0.2, 2.0, 100)[::-1]:
            resized = imutils.resize(gray, width= int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            #__if resized image smaller than template, then break the loop
            if  resized.shape[0] < tH or resized.shape[1] < tW:
                break

            #__preprocessing resized image and then apply template matching
            ##Cannyedge
            #edged = cv2.Canny(resized, 50, 200)
            ##Otsu threshold
            #ret_img,resized = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ##Simple threshold
            #ret_img, resized = cv2.threshold(resized,127,255,cv2.THRESH_BINARY)
            ##Adaptive threshold value is the mean of neighbourhood area
            #resized = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
            #Adaptive threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
            resized = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            (_,maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if visualize == True:
                #clone = np.dstack([edged, edged, edged])
                clone = np.dstack([resized, resized, resized])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0,0,255), 2)
                cv2.imshow("Visualizing", clone)
                cv2.waitKey(0)
                
            if maxVal > self.GetTemplate_Threshold():
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
                temp = [startX, startY, endX, endY]
                boundingBoxes = np.append(boundingBoxes, [temp], axis=0)
                max_value_list = np.append(max_value_list, [maxVal], axis=0)
                print("Max val = {} location {}".format(maxVal, temp))
                

        #__if detected on this scale size
        if len(boundingBoxes) > 1:
            boundingBoxes = np.delete(boundingBoxes, 0, axis=0)
            pick = self.non_max_suppression_slow(boundingBoxes, self.GetNMS_Threshold())
            print(len(pick))
            print(pick)
            for (startX, startY, endX, endY) in pick:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return image, boundingBoxes, max_value_list
            




def main():
    for imagePath in glob.glob(args["images"] + "/*.png"):
        
        template_loc_LPMQ = "/home/mhbrt/Desktop/Wind/Multiscale/temp5.png"
        nun_sukun_LPMQ = Marker(template_loc=template_loc_LPMQ, image_loc=imagePath, template_thresh = 0.7, nms_thresh = 0.3 )
        (result, bounding_box, max_value) = nun_sukun_LPMQ.Match_Template(visualize=False)
        cv2.imshow("Match Result", result)
        # if max_value != []:
        #     print(max_value)
        ##continue next file if not detected
        # if max_value == []:
        #     continue
        # print(max_value)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':main()
