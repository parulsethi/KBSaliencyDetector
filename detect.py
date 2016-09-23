import skimage
from skimage import color,io
from scipy import ndimage as ndi
import numpy as np

def kbdetect(image, scales):
    """ Detect keypoints based on Shannon entropy and weigh
        them with difference of descriptors in scale space
    """
    image = skimage.img_as_ubyte(image)
    nr, nc = image.shape
    # find pixels that we are going to examine
    mask = np.ones((nr, nc))
    r, c = np.nonzero(mask)
    nPix = len(r)

    nScales = len(scales)
    intensity_edges = [_ for _ in range(0, 257, 16)]

    previous_h = np.zeros((len(intensity_edges)-1, nPix))
    weights = np.zeros((nScales, nPix))
    entropy = np.zeros((nScales, nPix))

    # iterate through every possible region
    # iterate through scales
    for s_count in range(nScales):
        scale_size = scales[s_count]
        radius = int((scale_size)/2)

        # iterate through pixels
        for i in range(nPix):
            min_r, max_r = r[i]-radius, r[i]+radius
            min_c, max_c = c[i]-radius, c[i]+radius

            if min_r < 0:
                min_r = 0
            if max_r > nr:
                max_r = nr
            if min_c < 0:
                min_c = 0
            if max_c > nc:
                max_c = nc

            # compute the histogram of intensity values in this region
            patch = image[min_r:max_r, min_c:max_c]
            h = np.histogram(patch, bins=intensity_edges)[0]
            h = np.array([_/sum(h) for _ in h])

            # index of histogram values greater than zero
            idx = np.nonzero(h > 0)
            entropy[s_count, i] = -sum(h[idx]*np.log(h[idx]))

            if s_count >= 1:
                # first derivative in entropy space to calculate weights
                dif = abs(h - previous_h[:, i])
                factor = scales[s_count]**2/(2*scales[s_count]-1)
                weights[s_count, i] = factor * sum(dif)
                if s_count == 1:
                    weights[s_count-1, i] = weights[s_count, i]
            previous_h[:, i] = h

    # find local maxima in scale space by selecting second derivatives that are less than zero
    # second derivative kernel
    fxx = np.array([1, -2, 1])
    # evaluate second derivative using the kernel
    d_entropy = np.transpose(ndi.correlate1d(entropy.T, fxx, mode='nearest'))
    d_entropy[0, :] = 0
    int_pts = np.nonzero(d_entropy < 0)

    # weighting interest points by scale size times the first derivative of the scale-space function
    gamma = entropy[int_pts] * weights[int_pts]
    scale = scales[int_pts[0]]
    row = r[int_pts[1]]
    column = c[int_pts[1]]

    return np.array([gamma, scale, row, column])