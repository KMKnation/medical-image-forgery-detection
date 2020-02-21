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
    # print('-----------------')
    # cv2.imshow('filtered_image', filtered_img)
    # colored = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    # print(colored)
    # cv2.imshow('colored', colored)
    return colored


'''
CASIA 1 database contains 800 authentic andh
921 forged images.

the size
is 384  256 pixels.

'''

CASIA_ONE_AUTHENTIC_PATH = 'casia-dataset/CASIA1/Au/'
CASIA_ONE_FORGED_PATH = 'casia-dataset/CASIA1/Sp/'

'''
The CASIA 2 database
contains more than 7400 authentic and 5000
forged images. The images are in either JPEG,
TIFF, or BMP format.

'''

CASIA_TWO_AUTHENTIC_PATH = 'casia-dataset/CASIA2/Au/'
CASIA_TWO_FORGED_PATH = 'casia-dataset/CASIA2/Tp/'


# new_num_arr = np.load('dataset/np_casia_one_au.npy') # load

def prepare_casia_one_dataset():
    casia_one_au_arr = []
    casia_one_forged_arr = []

    # np.save('data.npy', num_arr) # save
    for image in os.listdir(CASIA_ONE_AUTHENTIC_PATH):
        imagepath = os.path.join(CASIA_ONE_AUTHENTIC_PATH, image)
        cv_image = cv2.imread(imagepath)
        print(str(image) + 'processing...')
        h, w = cv_image.shape[:2]
        if h != 256 and w != 384:
            continue
            # cv_image = imutils.resize(cv_image, width=384, height=256)
        if h == 256 and w == 384:
            processed_image = preprocess_image(cv_image)
            casia_one_au_arr.append(np.array(processed_image))
        else:
            print('Dimention mismatch')

    np_casia_one_au = np.array(casia_one_au_arr)
    np.save('dataset/np_casia_one_au.npy', np_casia_one_au)  # save
    print('CASIA1 Authentic Data Processed..')
    gc.collect()

    for image in os.listdir(CASIA_ONE_FORGED_PATH):
        imagepath = os.path.join(CASIA_ONE_FORGED_PATH, image)
        cv_image = cv2.imread(imagepath)
        print(str(image) + 'processing...')
        h, w = cv_image.shape[:2]
        if h != 256 and w != 384:
            continue
            # cv_image = imutils.resize(cv_image, width=384, height=256)
        if h == 256 and w == 384:
            processed_image = preprocess_image(cv_image)
            casia_one_forged_arr.append(np.array(processed_image))
        else:
            print('Dimention mismatch')

    np_casia_one_forged = np.array(casia_one_forged_arr)
    np.save('dataset/np_casia_one_forged.npy', np_casia_one_forged)  # save
    print('CASIA1 Forged Data Processed..')
    gc.collect()


def prepare_casia_two_dataset():
    casia_two_au_arr = []
    casia_two_forged_arr = []

    # np.save('data.npy', num_arr) # save
    for image in os.listdir(CASIA_TWO_AUTHENTIC_PATH):
        imagepath = os.path.join(CASIA_TWO_AUTHENTIC_PATH, image)
        cv_image = cv2.imread(imagepath)
        try:
            print(str(image) + 'processing...')
            h, w = cv_image.shape[:2]
            if h != 256 and w != 384:
                continue
                # cv_image = imutils.resize(cv_image, width=384, height=256)
            if h == 256 and w == 384:
                processed_image = preprocess_image(cv_image)
                casia_two_au_arr.append(np.array(processed_image))
            else:
                print('Dimention mismatch')
        except Exception as err:
            print(err)

    np_casia_two_au = np.array(casia_two_au_arr)
    np.save('dataset/np_casia_two_au.npy', np_casia_two_au)  # save
    print('CASIA2 Authentic Data Processed..')
    gc.collect()

    for image in os.listdir(CASIA_TWO_FORGED_PATH):
        imagepath = os.path.join(CASIA_TWO_FORGED_PATH, image)
        cv_image = cv2.imread(imagepath)
        try:
            print(str(image) + 'processing...')
            h, w = cv_image.shape[:2]
            if h != 256 and w != 384:
                continue
                # cv_image = imutils.resize(cv_image, width=384, height=256)
            if h == 256 and w == 384:
                processed_image = preprocess_image(cv_image)
                casia_two_forged_arr.append(np.array(processed_image))
            else:
                print('Dimention mismatch')
        except Exception as err:
            print(err)


    np_casia_two_forged = np.array(casia_two_forged_arr)
    np.save('dataset/np_casia_two_forged.npy', np_casia_two_forged)  # save
    print('CASIA2 Forged Data Processed..')
    gc.collect()


prepare_casia_one_dataset()
prepare_casia_two_dataset()
