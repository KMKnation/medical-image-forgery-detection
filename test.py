# coding=utf-8
# pip install scikit-image
from skimage import color, data, restoration
import cv2
import numpy as np
from skimage.restoration import estimate_sigma

def weiner_noise_reduction(img):
    # data.astronaut()
    img = color.rgb2gray(img)
    from scipy.signal import convolve2d
    psf = np.ones((5, 5)) / 25
    img = convolve2d(img, psf, 'same')
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    deconvolved_img = restoration.wiener(img, psf, 1100)

    return deconvolved_img

def estimate_noise(img):
    # img = cv2.imread(image_path)
    return estimate_sigma(img, multichannel=True, average_sigmas=True)


# image = cv2.imread('00000017_001_small.png')
image = cv2.imread('00000017_001_small.png')



'''
Step 1: If the image is a color image, decompose
it into red, green, and blue channels. If
the image is a monochrome image, there is
no need for this step.

'''
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('input', image)

enoise = estimate_noise(image)
print("Estimated Gaussian noise standard deviation = {}".format(enoise))
'''

Step 2: The Wiener-filter is applied to each
component of the color image or the monochrome
image itself. The output of this step
is an image (or component) free from noise.

'''

noise_free_image = weiner_noise_reduction(image)
cv2.imshow('noise_free_image', noise_free_image)


'''

Step 3: The noise-free image is subtracted from
the original image to get an estimated noise
pattern of the image. The noise pattern is
considered as the fingerprint of the image.
If any forgery is done, this fingerprint is distorted.

'''
import random
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


noise_img = sp_noise(image,0.05)
cv2.imshow('noise_img', noise_img)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
fingerprint = gray - noise_free_image
cv2.imshow('fingerprint', fingerprint)

'''

Step 4: The multi-resolution regression filter is
applied to the noise pattern. The regression
filter is illustrated in Fig. 3. In this filter, the
nearest eight-pixel positions have weight 1,
while the next neighboring pixels’ positions
have weight 2, and so on. The characteristic
of this filter is to capture the relative intensity
of a center pixel. The final weight is normalized
between 0 and 255 to maintain the
intensity level of a gray image. 

Here, we have used median.

'''

from skimage.filters import median
fingerprint = fingerprint/255
# filtered_img = median(fingerprint, selem=None, out=None, mask=None, shift_x=False,
#            shift_y=False, mode='nearest', cval=0.0, behavior='rank')
filtered_img = median(fingerprint, selem=None, out=None, mask=None, shift_x=False,
           shift_y=False)

print(filtered_img.shape)
cv2.imshow('filtered', filtered_img)
cv2.waitKey(0)
