import numpy as np
# %KBPRUNE Remove weak regions from Kadir-Brady candidates

  # Now we sort the gamma values in temp before we get the distances,
  # instead of the other way around

def kbprune(candidates,K,v_th):
    cgamma = candidates[0]
    cscale = candidates[1]
    cr = candidates[2]
    cc = candidates[3]

    # apply a global threshold to the gamma vals
    thresh_val = .6 * max(cgamma)

    # indices = find(cgamma > thresh_val)
    indices = np.nonzero(cgamma > thresh_val)

    tgamma = cgamma[indices]
    tscale = cscale[indices]
    tr = cr[indices]
    tc = cc[indices]

    # sort the gamma values, and order everything by that
    sidx = np.argsort(tgamma)[::-1]
    tgamma = tgamma[sidx]
    tscale = tscale[sidx]
    tr = tr[sidx]
    tc = tc[sidx]

    # create a distance matrix
    n = max(tgamma.shape)

    # if K+1 > n
    #     regions.gamma = temp.gamma(1)
    #     regions.r = temp.r(1)
    #     regions.c = temp.c(1)
    #     regions.scale = temp.scale(1)
    #     return
    # end

    D = np.zeros((n,n))

    pts = [tc,tr,tscale]

    # fill it with distances
    for i in range(1,n):
        pt = pts[i,:]

        dists = np.sqrt(((pts-np.tile(pt,(size(pts,1),1)))**2).sum(axis=1))

        D[i,:] = dists.T
        D[:,i] = dists        

    nReg = 0
    regions = []
    pos = np.zeros((3,K+1))

    # now do the pruning process
    for i in range(1,n):
        index = i

        pos[1,1] = tc[index]
        pos[2,1] = tr[index]
        pos[3,1] = tscale[index]

        [sD,sidx] = sort(D[index,:])

        for j in range(1,K):
            pos[1,j+1] = tc[sidx(j+1)]
            pos[2,j+1] = tr[sidx(j+1)]
            pos[3,j+1] = tscale[sidx(j+1)]
        
        cent = np.mean(pos,axis=1)

        v = np.var(np.sqrt(((pos-np.tile(cent,([1,K+1]))**2).sum(axis=0))))

        if v > v_th:
            continue

        # now that we know the regions is "suffiently clustered", make sure the
        # region is "far enough" from already clustered regions
        if nReg > 0:
            d = np.sqrt((([rc,rr,rscale] - np.tile(cent.T,(nReg,1)))**2).sum(axis=1))
            if (cent[3]>=d).sum() == 0:
                nReg = nReg+1
                rc[nReg,1] = cent[1]
                rr[nReg,1] = cent[2]
                rscale[nReg,1] = cent[3]
                rgamma[nReg,1] = tgamma[index]
        else:
            nReg = nReg+1
            rc[nReg,1] = cent[1]
            rr[nReg,1] = cent[2]
            rscale[nReg,1] = cent[3]
            rgamma[nReg,1] = tgamma[index]