import mess 
import cv2
import glob

dot = mess.FontWrapper(image_loc='',
                       loc_list='',
                       thresh_list={'': ''},
                       image='')

for image_file in glob.glob('/home/mhbrt/Desktop/Wind/Multiscale/temp/marker/one/*.png'):
    image = cv2.imread(image_file)
    cv2.imshow('df', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(dot.dot_checker(gray))
