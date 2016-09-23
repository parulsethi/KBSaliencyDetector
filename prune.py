import numpy as np
from scipy import ndimage as ndi

def kbprune(candidate_regions, saliency_threshold, v_th, K=7):
    """ Pruning of detected keypoints and selecting 
        the ones with high saliency score
    """
    gamma, scale, row, column = (np.array([]) for i in range(4))
    cgamma, cscale, cr, cc = candidate_regions

    # apply a global threshold to the gamma values
    threshold = saliency_threshold * max(cgamma)
    t = np.nonzero(cgamma > threshold)
    t_gamma, t_scale, t_r, t_c = cgamma[t], cscale[t], cr[t], cc[t]

    # sort the gamma values and order the rest by that
    s_i = np.argsort(t_gamma)[::-1]
    t_gamma, t_scale, t_r, t_c = t_gamma[s_i], t_scale[s_i], t_r[s_i], t_c[s_i]

    # create a Distance matrix
    n = max(t_gamma.shape)
    D = np.zeros((n, n))
    pts = np.array(list(zip(t_c, t_r, t_scale)))

    # fill it with distances
    for i in range(n):
        pt = pts[i][:]
        # calculate the distances b/w regions
        dists = np.sqrt(((pts-np.tile(pt, (pts.shape[0], 1)))**2).sum(axis=1))
        D[i, :], D[:, i] = dists.T, dists

    nReg = 0
    # clusters matrix
    cluster = np.zeros((3, K+1))

    # pruning process
    for index in range(n):
        cluster[0, 0] = t_c[index]
        cluster[1, 0] = t_r[index]
        cluster[2, 0] = t_scale[index]
        s_i = np.argsort(D[index, :])
        # fill in the neighbouring regions
        for j in range(K):
            cluster[0, j+1] = t_c[s_i[j+1]]
            cluster[1, j+1] = t_r[s_i[j+1]]
            cluster[2, j+1] = t_scale[s_i[j+1]]

        # clusters center point
        center = np.array([np.mean(cluster, axis=1)])

        # check if the regions are "suffiently clustered", if variance is less than threshold
        v = np.var(np.sqrt(((cluster - np.tile(center.T, (1, K+1)))**2).sum(axis=0)))
        if v > v_th:
            continue

        center = np.mean(cluster, axis=1)
        if nReg > 0:
            # make sure the region is "far enough" from already clustered regions
            d = np.sqrt(((np.array(list(zip(column, row, scale)))
                            - np.tile(center.T, (nReg, 1)))**2).sum(axis=1))
            if (center[2] >= d).sum() == 0:
                nReg = nReg+1
                column = np.append(column, center[0])
                row = np.append(row, center[1])
                scale = np.append(scale, center[2])
                gamma = np.append(gamma, t_gamma[index])
        else:
            nReg = nReg+1
            column = np.append(column, center[0])
            row = np.append(row, center[1])
            scale = np.append(scale, center[2])
            gamma = np.append(gamma, t_gamma[index])

    return np.array([row, column, scale])
    