import os
import cv2
import gc
from skimage import color, data, restoration
import cv2
import numpy as np
from skimage.restoration import estimate_sigma
from skimage.filters import median
import config
import imutils

img_height = 256
img_width = 384

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


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    enoise = estimate_noise(image)
    noise_free_image = weiner_noise_reduction(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fingerprint = gray - noise_free_image
    fingerprint = fingerprint / 255
    filtered_img = median(fingerprint, selem=None, out=None, mask=None, shift_x=False,
                          shift_y=False, mode='nearest', cval=0.0, behavior='rank')
    colored = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    return colored


DDSM_DATASET = 'CDDSM/figment.csee.usf.edu/pub/DDSM/cases'

NORMAL = os.path.join(DDSM_DATASET, 'normals')
ABNORMAL = os.path.join(DDSM_DATASET, 'cancers')

import glob
import random
normalGlob = glob.glob(NORMAL+"/*/*/*.jpg")
abNormalGlob = glob.glob(ABNORMAL+"/*/*/*.jpg")

def prepare_ddsm_dataset():
    casia_one_au_arr = []
    casia_one_forged_arr = []

    # np.save('data.npy', num_arr) # save
    for imagepath in normalGlob:
#         imagepath = os.path.join(CASIA_ONE_AUTHENTIC_PATH, image)
        cv_image = cv2.imread(imagepath)
        cv_image = cv2.resize(cv_image, (img_width, img_height))
        print(str(imagepath) + 'processing...')
        processed_image = preprocess_image(cv_image)
        casia_one_au_arr.append(np.array(processed_image))

    np_casia_one_au = np.array(casia_one_au_arr)
    np.save('CDDSM/np_ddsm_normal.npy', np_casia_one_au)  # save
    print('DDSM Authentic Data Processed..')
    gc.collect()

    for imagepath in abNormalGlob:
        cv_image = cv2.imread(imagepath)
        cv_image = cv2.imread(imagepath)
        cv_image = cv2.resize(cv_image, (img_width, img_height))
        print(str(imagepath) + 'processing...')
        processed_image = preprocess_image(cv_image)
        casia_one_forged_arr.append(np.array(processed_image))

    np_casia_one_forged = np.array(casia_one_forged_arr)
    np.save('CDDSM/np_ddsm_abnormal.npy', np_casia_one_forged)  # save
    print('DDSM Forged Data Processed..')
    gc.collect()

prepare_ddsm_dataset()
