import numpy as np
# %KBPRUNE Remove weak regions from Kadir-Brady candidates

  # Now we sort the gamma values in temp before we get the distances,
  # instead of the other way around

def kbprune(candidates,K,v_th):

    rgamma = np.array([])
    rscale = np.array([])
    rr = np.array([])
    rc = np.array([])

    cgamma = candidates[0]
    cscale = candidates[1]
    cr = candidates[2]
    cc = candidates[3]

    # apply a global threshold to the gamma vals
    thresh_val = .7 * max(cgamma)

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

    pts = np.array(list(zip(tc,tr,tscale)))


    # fill it with distances
    for i in range(n):
        pt = pts[i][:]

        dists = np.sqrt(((pts-np.tile(pt,(pts.shape[0],1)))**2).sum(axis=1))

        D[i,:] = dists.T
        D[:,i] = dists

    nReg = 0
    regions = []
    pos = np.zeros((3,K+1))

    # now do the pruning process
    for i in range(n):
        index = i

        pos[0,0] = tc[index]
        pos[1,0] = tr[index]
        pos[2,0] = tscale[index]

        sidx = np.argsort(D[index,:])

        for j in range(K):
            pos[0,j+1] = tc[sidx[j+1]]
            pos[1,j+1] = tr[sidx[j+1]]
            pos[2,j+1] = tscale[sidx[j+1]]

        cent = np.array([np.mean(pos,axis=1)])

        v = np.var(np.sqrt(((pos-np.tile(cent.T,(1,K+1)))**2).sum(axis=0)))

        # if v > v_th:
        #     continue
        # print("zzzzzzzzzzzzzzzzzzzzzzzz")
        # now that we know the regions is "suffiently clustered", make sure the
        # region is "far enough" from already clustered regions
        cent = np.mean(pos,axis=1)

        if nReg > 0:
            pp = np.array(list(zip(rc,rr,rscale)))
            d = np.sqrt(((pp - np.tile(cent.T,(nReg,1)))**2).sum(axis=1))
            if (cent[2]>=d).sum() == 0:
                nReg = nReg+1
                rc = np.append(rc,cent[0])
                rr = np.append(rr,cent[1])
                rscale = np.append(rscale,cent[2])
                rgamma = np.append(rgamma,tgamma[index])
        else:
            nReg = nReg+1
            # print(cent[0],cent[1],cent[2])
            rc = np.append(rc,cent[0])
            rr = np.append(rr,cent[1])
            rscale = np.append(rscale,cent[2])
            rgamma = np.append(rgamma,tgamma[index])


    # print("tscale",tscale)
    return rgamma,rscale,rr,rc
