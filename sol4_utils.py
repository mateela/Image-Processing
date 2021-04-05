from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy import signal
from scipy.ndimage.filters import convolve

GRAYSCALE_REP = 1
RGB_REP = 2
LAST_BIN = 255


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: - representation code, either 1 or 2 defining whether the output
    should be a grayscale image (1) or an RGB image (2).
    :return: an image, represented by a matrix of type np.float64 with intensities (either grayscale or
    RGB channel intensities) normalized to the range [0, 1].
    """

    # Input test & Reading the image:
    try:
        im = imread(filename)
    except FileNotFoundError:
        print("No Such File")
        exit(0)

    if representation not in (GRAYSCALE_REP, RGB_REP):
        representation = GRAYSCALE_REP     # Default representation

    # Grayscale representation:
    if representation == GRAYSCALE_REP:
        if len(im.shape) < 3:
            return int2float(im)
        else:
            return rgb2gray(im)

    # RGB representation:
    else:
        return int2float(im)


def int2float(im):
    """
    This function converts an image, represented by a matrix of type np.uint8 with intensities
    (either grayscale or RGB channel intensities) ranging from 0 to 255, to an image represented by a
    matrix of type np.float64 with intensities normalized to the range [0, 1]. This is an inplace change.
    :param im: the matrix representing the image.
    :return: the converted image.
    """
    return im.astype(np.float64, copy=False) / LAST_BIN


def float2int(im):
    """
    This function converts an image, represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities) normalized to the range [0, 1], to an image
    represented by a matrix of type np.uint8 with intensities ranging from 0 to 255. This is an inplace change.
    :param im: the matrix representing the image.
    :return: the converted image.
    """
    return np.round(im * LAST_BIN).astype(np.uint8, copy=False)


def generate_filter_vec(filter_size):
    """
    This function generates a filter vector. The filter vector is a row vector of shape (1, filter_size) used
    for the pyramid construction.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter).
    :return: a Gaussian filter which is a row vector of shape (1, filter_size).
    """

    if filter_size == 1:
        return np.array([1]).reshape((1, filter_size))

    vec = np.array([1, 1])
    while vec.shape[0] < filter_size:
        vec = signal.convolve(vec, np.array([1, 1]))
    return (vec * (1 / 2 ** (filter_size - 1))).reshape((1, filter_size))


def reduce(im, filter_vec):
    """
    This function reduces the size of an image. First, it blurs the image using the filter vector, and then
    omits every second pixel in every second row.
    :param im: a grayscale image with double values in [0, 1].
    :param filter_vec: a Gaussian filter which is a row vector of shape (1, filter_size).
    :return: a grayscale image with double values in [0, 1], and dimension: (original rows / 2, original columns / 2).
    """

    return convolve(convolve(im, filter_vec), filter_vec.T)[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function construct a Gaussian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter).
    :return: pyr - the Gaussian pyramid of the given image.
             filter_vec - the Gaussian filter which is a row vector of shape (1, filter_size).
    """

    filter_vec = generate_filter_vec(filter_size)
    pyr = [im]
    for i in range(max_levels - 1):

        if pyr[i].shape[0] / 2 < 16 or pyr[i].shape[1] / 2 < 16:
            # Ensuring that the minimum dimension (height or width) of the lowest resolution image in the
            # pyramid is not smaller than 16
            break

        pyr.append(reduce(pyr[i], filter_vec))

    return pyr, filter_vec


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

