import numpy as np
from pandas import read_csv

if __name__ == '__main__':
    # XYPT = np.loadtxt('/home/mahdi/Desktop/PhD_application/eth_additive_phd/data/Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer0002.csv')

    # XYPT = np.genfromtxt('/home/mahdi/Desktop/PhD_application/eth_additive_phd/data/Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer0002.csv', delimiter=',')

    dataframe = read_csv('/home/mahdi/Desktop/PhD_application/eth_additive_phd/data/Build Command Data/XYPT Commands/T500_3D_Scan_Strategies_fused_layer0002.csv')
    XYPT = dataframe.values
    print('end')