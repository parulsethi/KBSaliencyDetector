from detect import kbdetect
from prune import kbprune
from show import kbshow
from skimage import io,transform,color,data
import numpy as np
from PIL import Image

# def rgb2gray(rgb):
#
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989*r + 0.5870*g + 0.1140*b
#
#     return gray

im = io.imread('panda1.jpg')
im = color.rgb2gray(im)
# print(im[0:5,0:5])

row,column = im.shape
min_size = min(row,column)

if min_size == row:
    temp1 = 100/row
    temp2 = int(column*temp1)
    im = transform.resize(im,(100,temp2+1))
else:
    temp1 = 100/column
    temp2 = int(row*temp1)
    im = transform.resize(im,(temp2+1,100))

nr,nc = im.shape

# if nargin < 5
#     mask = ones(nr,nc);
# end

# if nargin < 4
#     v_th = 1;
# end

# if nargin < 3
#     K = 5;
# end

# if nargin < 2
#     windows = [10:2:25];
# end

K = 7
v_th = 9
# scales
windows = [_ for _ in range(10,25,2)]
mask = np.ones((nr,nc))

# the first step is to find local maxima in scale-space
base_regions = kbdetect(im,windows,mask)

# now clustering must be done
regions = kbprune(base_regions, K, v_th)

kbshow(im,regions)

# regions.gamma
