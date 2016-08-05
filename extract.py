function [regions] = kbextract(im,windows,K,v_th,mask)

from skimage import io,transform

import kbdetect

im = io.imread('cars\image_0006.jpg');
row,column = im.shape

min_size = min(row,column);

if min_size == row: 
    temp1,temp2 = 100/row,int(column*temp1)
    im = transform.resize(im,(100 temp2))
else:
    temp1,temp2 = 100/column,int(row*temp1)
    im = transform.resize(im,(temp2 100))

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

windows = [_ for _ in range(10,25,2)]
mask = np.ones((nr,nc))

# the first step is to find local maxima in scale-space
base_regions = kbdetect(im,windows,mask);

# now clustering must be done
regions = kbprune(base_regions, K, v_th);

kbshow(im,regions);

regions.gamma
