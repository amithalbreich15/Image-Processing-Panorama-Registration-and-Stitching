from scipy.signal import convolve2d
import numpy as np
import imageio as io
from skimage import color
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt


GRAYSCALE = 1
RGB = 2
RGB_CHANNELS = 3


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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    inter_img = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(inter_img, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(inter_img, vmin=0, vmax=1)
    plt.show()


def down_sampling(im, factor):
    """
    Performs downsample process on the image slicing from image and return
    a shrunken image according to factor. In our case it is shrunk by factor 2
    and skips every 2nd row and 2nd column elements.
    :param im: The image
    :param factor: factor (scalar) of the downsampling process
    :return: downsampled image (shrunk by factor and skips each 'factor'ish
    element in the original image - in our case skips each 2nd row and 2nd
    column elements).
    """
    return im[::factor, ::factor]


def up_sampling(im, factor):
    """
    Performs upsample process on the image expanding it by chosen factor and
    padding it alternately with zeros (when using factor 2).
    :param im: image to perform upsampling process on.
    :param factor: factor (scalar) of the upsampling process
    :return: upsampled image (expanded and padded with zeros alternately).
    """
    rows = len(im)
    cols = len(im[0])
    up_sampled_img = np.zeros((rows * factor, cols * factor))
    up_sampled_img[::factor, ::factor] = im
    return up_sampled_img


def get_convolve_vector(filter_size, norm_factor):
    """
    Helper function that gets the convolved filter vector to use in reduce and
    expand processes.
    :param filter_size: filter desired size
    :param norm_factor: normalize factor (scalar).
    :return: filter vector in desired length (filter size) after convolution
    process.
    """
    convolution_base = np.array([1, 1])
    if filter_size == 2:
        return convolution_base
    filter_vec = convolution_base
    while len(filter_vec) != filter_size:
        filter_vec = np.convolve(filter_vec, convolution_base)
    filter_vec_sum = np.sum(filter_vec)
    filter_vec = (filter_vec / filter_vec_sum) * norm_factor
    filter_vec = filter_vec.reshape(1, len(filter_vec))
    return filter_vec


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    img = np.array(io.imread(filename).astype(np.float64))
    if np.max(img.flatten()) > 1:
        img = img.astype(np.float64) / 255  # normalize the data to range [0,1]
    if representation == GRAYSCALE and len(np.shape(img)) == 3:
        img = color.rgb2gray(img)
    return img


def pad_along_axis(array: np.ndarray, target_length: int,
                   axis: int = 0) -> np.ndarray:
    """
    Helper function - pads image as ndarray with zeros along a chosen axis
    expands the original image with zeros on the right or bottom of the image.
    :param array: ndarray represents the image to pad.
    :param target_length: target length (width/height) of the padded image -
    after the padding process.
    :param axis: axis to pad the ndarray (image) in with zeros.
    :return: padded ndarray (images) with extra cols to the right or rows on
     the bottom according to users choice.
    """
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def linear_stretching(image):
    """
    performs linear stretching to a given image pixels.
    :param image: The image
    :return: Linear Stretched image
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    convolve_x = convolve(im, blur_filter, mode='constant')
    convolve_x_y = convolve(convolve_x, blur_filter.T, mode='constant')
    resized_image = down_sampling(convolve_x_y, 2)
    return resized_image


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    upsampled_img = up_sampling(im, 2)
    convolve_x = convolve(upsampled_img, np.dot(blur_filter, 2),
                          mode='constant')
    convolve_x_y = convolve(convolve_x, np.dot(blur_filter, 2).T,
                            mode='constant')
    return convolve_x_y


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = [im]
    resized_img = im
    filter_vec = get_convolve_vector(filter_size, 1)
    for i in range(max_levels - 1):
        if resized_img.shape[0] <= 16 or resized_img.shape[1] <= 16:
            break
        resized_img = reduce(resized_img, filter_vec)
        pyr.append(resized_img)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gpyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = [(np.subtract(gpyr[i], expand(gpyr[i + 1], filter_vec))) for i in
           range(len(gpyr) - 1)]
    pyr.append(gpyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    lpyr = [(coeff[i] * lpyr[i]) for i in range(len(lpyr))]
    for i in range(len(lpyr) - 1, 0, -1):
        lpyr[i - 1] += expand(lpyr[i], filter_vec)
    img = lpyr[0]
    return img


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    lpyr1, filter_vec_l1 = build_laplacian_pyramid(im1, max_levels,
                                                   filter_size_im)
    lpyr2, filter_vec_l2 = build_laplacian_pyramid(im2, max_levels,
                                                   filter_size_im)
    mask = mask.astype(np.float64)
    gpyr_mask, filter_vec_gpyr = build_gaussian_pyramid(mask, max_levels,
                                                        filter_size_mask)
    coeff = [1] * max_levels
    l_pyr_out = [gpyr_mask[k] * lpyr1[k] + ((1.0 - gpyr_mask[k]) * lpyr2[k])
                 for k in range(max_levels)]
    return np.clip((laplacian_to_image(l_pyr_out, filter_vec_l1, coeff)), 0, 1)


