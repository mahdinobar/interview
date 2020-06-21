import numpy as np
from PIL import Image
from skimage.measure import perimeter
from skimage.filters import threshold_otsu
import os


if __name__ == '__main__':
    # extract three features of layers 1 to 250 of MPM camera frames
    for l in range(1, 251):
        dir = "/home/mahdi/Desktop/PhD_application/eth_additive_phd/data/In-situ Meas Data/Melt Pool Camera/MIA_L{:0>4d}".format(2)
        list = os.listdir(dir)  # dir is your directory path
        number_frames = len(list)
        features = np.zeros((number_frames, 3))

        # for MPM frames of layer l
        for f in range(1, number_frames+1):
            img = Image.open(dir+"/frame{:0>5d}.bmp".format(f))
            img = np.array(img)
            # consider 10 threshold of non melted part
            img = img * (img > 10)

            # consider values higher than 20 to belong to higher temparutre area: nice measure for speed: higher values implies higher speed
            surface_feature = np.sum(img>20)

            # consider values higher than 240 to belong to core melt pool: nice measure for power: higher value implies more power
            core_surface_feature = np.sum(img>240)

            # compacity feature: nice measure of compactness of the object(entire melt pool): higher implies near the borders
            #find otsu optimum threshold for the resultant image
            try:
                thresh = threshold_otsu(img)
                binary = img > thresh
                P = perimeter(binary, neighbourhood=4)
                A = np.sum(binary)
                compacity_feature = (P ** 2) / A
            except:
                compacity_feature = 0.

            features[f-1, :] = [surface_feature, core_surface_feature, compacity_feature]
        # save table of features for layer l at ./outputs/question_3/
        np.savetxt('/home/mahdi/Desktop/PhD_application/eth_additive_phd/interview/outputs/question_3/features_layer{:0>4d}.csv'.format(l), features)
        print('layer {} features table saved!'.format(l))



