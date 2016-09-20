from detect import kbdetect
from prune import kbprune
from skimage import io,transform,color,data
import numpy as np

def extract_keypoints(image, min_scale=10, max_scale=25, saliency_threshold=0.6, clustering_threshold=7):
    # scales for keypoints
    scales = np.array([_ for _ in range(min_scale, max_scale, 2)])
    # detect keypoints in scale-space
    base_regions = kbdetect(image, scales)
    # pruning based on thresholds
    regions = kbprune(base_regions, saliency_threshold, clustering_threshold, K=7)
    # returns (y,x,scale)
    regions.T

def show(im,regions):
    regions = regions.T
    rs = regions[2]
    rr = regions[0]
    rc = regions[1]

    fig,ax = plt.subplots()
    for y,x,r in zip(rr,rc,rs/2):
        c = plt.Circle((x,y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    ax.imshow(im, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    image = data.astronaut()[0:300, 0:300]
    regions = extract_keypoints(color.rgb2gray(image))
    show(image,regions)