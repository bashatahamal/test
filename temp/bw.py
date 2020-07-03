import cv2
import glob
import numpy as np 
from matplotlib import pyplot as plt

def horizontal_projection(image_h):
    image = image_h.copy()
    image[image < 127] = 1
    image[image >= 127] = 0
    h_projection = np.sum(image, axis=1)

    return h_projection

for imagePath in sorted(glob.glob("*.jpg")):
    print('________________Next File_________________')
    image = cv2.imread(imagePath)
    original_image = cv2.imread(imagePath)
    h, w, _ = original_image.shape
    if h > 1200 :
        original_image = cv2.resize(original_image, (int(0.5 * w), int(0.5 * h)))
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)
    
    
    
    img = cv2.imread(imagePath,0)

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    magnitued, angle = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])

    print(angle.shape)

    dft_shift = np.fft.fftshift(dft)
    m, a = cv2.cartToPolar(np.real(dft_shift), np.imag(dft_shift))
    # print(a)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # template = cv2.Canny(gray, 50, 200)
    # Otsu threshold
    ret_img, image1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY
                                   + cv2.THRESH_OTSU)
    
    ret_img, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY
                                   + cv2.THRESH_OTSU)
    plt.figure(1)
    body_v_proj = horizontal_projection(image1)
    plt.subplot(121), plt.imshow(image1)
    plt.subplot(122), plt.plot(
        body_v_proj, np.arange(0, len(body_v_proj), 1)
    )
    # plt.show()
    from scipy.signal import find_peaks
    plt.figure(2)
    x = body_v_proj*-1
    peaks, _ = find_peaks(x, distance=50)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()
    cv2.waitKey(0)

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make
# it positive
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    # cv2.imshow("Input", image)
    # cv2.imshow("Rotated", rotated)
    # cv2.waitKey(0)
    
    # Simple threshold
    # ret_img, image2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Adaptive threshold value is the mean of neighbourhood area
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)
    # Adaptive threshold value is the weighted sum of neighbourhood
    # values where weights are a gaussian window
    # image = cv2.adaptiveThreshold(gray, 255,
    #                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow('otsu', image1)
    # cv2.imshow('simple', image2)
    # cv2.imshow('adapt mean', image3)
    # cv2.imshow('adapt gaussian', image)
    # cv2.waitKey(0)
    # image = cv2.bitwise_not(image)
    # kernel = np.ones((1,1), np.uint8)
    # dilation = cv2.dilate(final_img.copy(),kernel,iterations = 1)
    # kernel = np.ones((2,2), np.uint8)
    # image = cv2.erode(image,kernel,iterations = 1)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # image = cv2.bitwise_not(image)
    # closing = cv2.morphologyEx(final_img.copy(), cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('morph', image1)
    # print('morph')
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
