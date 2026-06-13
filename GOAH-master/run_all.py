import os
import torch
import warnings
import numpy as np
import random

from model8 import GOAH
from Quaternion_MLdata_load2 import MLload_data

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    name_all = 'wine heart breast aa mm ttt glass yeast iris'
    name_list = name_all.split()
    for name_id in range(9):
        seed = 125025
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        name = name_list[name_id]
        method_name = 'GOAH'
        features, adjacency, labels = MLload_data(name)

        layers = [512, 256, 128]
        acts = [torch.nn.functional.relu] * len(layers)
        learning_rate = 10 ** -4 * 4
        pretrain_learning_rate = 0.0001
        lamSC = np.power(2.0, 1)
        coeff_reg = 0.0001

        max_epoch = 100
        max_iter = 5
        pre_iter = 5

        decomposition = 'symeig' # symeig,svd,eigh
        is_norm = False

        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        for _ in range(10):
            gae = GOAH(name, features, adjacency, labels,
                       decomposition=decomposition, is_norm=is_norm,
                       layers=layers, acts=acts, max_epoch=max_epoch, max_iter=max_iter,
                       coeff_reg=coeff_reg, learning_rate=learning_rate,
                       seed=seed, lam=lamSC,
                       hyper_n_clusters=20, hyper_S=5)

            gae.cuda()

            gae.pretrain(pre_iter, learning_rate=pretrain_learning_rate)
            acc, nmi, ari, f1 = gae.run()

            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)


        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)

        print('ACC:', acc_list.mean(), "±", acc_list.std())
        print('NMI:', nmi_list.mean(), "±", nmi_list.std())
        print('ARI:', ari_list.mean(), "±", ari_list.std())
        print('F1 :', f1_list.mean(), "±", f1_list.std())
        print("\n")



