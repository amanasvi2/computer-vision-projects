#!/usr/bin/python3

import numpy as np


def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    m, n = image.shape
    k, j = filter.shape
    pad_height = (m-k) // 2
    pad_width = (n-j) // 2
    padded_filter = np.pad(filter, ((pad_height, m - k - pad_height), (pad_width, n - j - pad_width)), mode='constant')
    image_freq = np.fft.fft2(image)
    filter_freq = np.fft.fft2(padded_filter)
    conv_result_freq = image_freq * filter_freq
    conv_result = np.fft.ifft2(conv_result_freq).real
    conv_result = np.fft.fftshift(conv_result)

    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, conv_result_freq, conv_result 


def my_deconv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation.
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the deconvolution in the frequency domain, and 
    - the result of the deconvolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        deconv_result_freq: array of shape (m, n)
        deconv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the deconvolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 to see what this means and to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    m, n = image.shape
    k, j = filter.shape
    pad_height = (m-k) // 2
    pad_width = (n-j) // 2
    padded_filter = np.pad(filter, ((pad_height, m - k - pad_height), (pad_width, n - j - pad_width)), mode='constant')
    image_freq = np.fft.fft2(image)
    filter_freq = np.fft.fft2(padded_filter)
    deconv_result_freq = image_freq / filter_freq
    deconv_result = np.fft.ifft2(deconv_result_freq).real
    deconv_result = np.fft.fftshift(deconv_result)

    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, deconv_result_freq, deconv_result





