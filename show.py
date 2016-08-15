# KBSHOW Display regions found by the Kadir-Brady algorithm

import matplotlib.pyplot as plt

def kbshow(im,regions):

    rg = regions[0]
    rs = regions[1]
    rr = regions[2]
    rc = regions[3]
    # theta = [_ for _ in np.arange(-0.03,2*(np.pi),0.01)]

    # for i in range(1:len(rg)):
    #     r = regions.scale[i]/2

    #     x = r*np.cos(theta)
    #     y = r*np.sin(theta)

    #     X = x+regions.c[i]
    #     Y = y+regions.r[i]
        
    #     j=num2str(i)

    #     plot(X,Y,'r-')
    #     text(regions.c(i),regions.r(i),j,'Fontsize',18,'Color',[1 1 0])

    fig,ax = plt.subplots()
    for y,x,r in zip(rr,rc,rs/2):
        c = plt.Circle((x,y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    ax.imshow(im, interpolation='nearest')
    plt.show()

    # # Plot six regions with highest saliency
    # figure(2)
    # subplot('position', [0 0 1 1])
    # imagesc(im)
    # colormap gray
    # hold on

    # theta = [-.03:.01:2*pi]

    # for i=1:6
    #     r = regions.scale(i)/2

    #     x = r*cos(theta)
    #     y = r*sin(theta)

    #     X = x+regions.c(i)
    #     Y = y+regions.r(i)
        
    #     j=num2str(i)

    #     plot(X,Y,'r-')
    #     text(regions.c(i),regions.r(i),j,'Fontsize',18,'Color',[1 1 0])
    # end


    # index=1

    # figure(3)
    # for i=1:6
    #     xmin=regions.c(i)-(regions.scale(i)/2)
    #     ymin=regions.r(i)-(regions.scale(i)/2)
    #     height=regions.scale(i)
    #     width=regions.scale(i)
    #     cropped_image=imcrop(im,[xmin ymin width height])
    #     resized_image=imresize(cropped_image, [11 11])
    #     subplot(1,6,i), imshow(resized_image)
    #     index=index+1
    # end


    # return