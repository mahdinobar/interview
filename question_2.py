import numpy as np
from pandas import read_csv

if __name__ == '__main__':

    # read data from layer 2 to layer 161
    for l in range(2, 162):
        # read data of layer l
        dataframe = read_csv(
            '/home/mahdi/Desktop/PhD_application/eth_additive_phd/data/Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer{:0>4d}.csv'.format(
                l))
        XYPT = dataframe.values
        # extract camera triggered data
        arg_r_now = np.argwhere(XYPT[:, 3] == 2).squeeze()
        arg_r_past = (np.argwhere(XYPT[:, 3] == 2) - 1).squeeze()
        # extract camera triggered laser positions and power
        XYP = XYPT[arg_r_now, :3]
        time_stamp = 10e-6
        # extract camera triggered approximate velocity
        V = np.linalg.norm((XYP[:, :2] - XYPT[arg_r_past, :2]) / (time_stamp), axis=1)
        # save data into a table
        np.savetxt(
            '/home/mahdi/Desktop/PhD_application/eth_additive_phd/interview/outputs/question_2/XYPV_layer{:0>4d}.csv'.format(l),
            np.hstack([XYP, V.reshape(V.shape[0], 1)]))
        print('layer {} XYPV table saved!'.format(l))
