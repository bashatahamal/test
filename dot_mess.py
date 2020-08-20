import mess
import cv2

dot = mess.FontWrapper(image_loc='',
                       loc_list='',
                       thresh_list={'': ''},
                       image='')

image = cv2.imread('/home/mhbrt/Downloads/0_868.png')
#cv2.imshow('df', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(dot.dot_checker(gray))
