import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import os




def find_NaN_auto(data, minK, maxK):

    minK = minK
    maxK = maxK
    row = data.shape[0]

    nbrs = NearestNeighbors(n_neighbors=maxK, algorithm='ball_tree').fit(data)
    knn_dis, knn_indices = nbrs.kneighbors(data)

    for i in range(row):
        if knn_indices[i, 0] != i:
            if i in knn_indices[i, :]:
                index = np.argwhere(knn_indices[i, :] == i)
                index = index[0][0]
            else:
                continue

            knn_indices[i, 1:index + 1] = knn_indices[i, 0:index]
            knn_indices[i, 0] = i

    NaNeighbor_map = np.zeros((row, row))
    NaN_distance = np.zeros((row, row))

    K = minK
    for i in range(row):
        i_knn = knn_indices[i, :K]
        for j in range(1, K):
            i_jnn = i_knn[j]
            j_knn = knn_indices[i_jnn, :K]
            if i in j_knn:
                NaNeighbor_map[i, i_jnn] = 1
                NaNeighbor_map[i_jnn, i] = 1
                NaN_distance[i, i_jnn] = knn_dis[i, j]
                NaN_distance[i_jnn, i] = knn_dis[i, j]

    zero_num = 0
    for i in range(row):
        if np.sum(NaNeighbor_map[i, :]) == 0:
            zero_num += 1

    if zero_num == 0:
        return NaNeighbor_map, NaN_distance

    Flag = True

    count_old = zero_num
    repeat_time = 0
    while Flag:
        if K >= maxK:
            Flag = False
            break

        next_K = K + 1
        for i in range(row):
            i_k_n = knn_indices[i, next_K - 1]
            i_k_n_knn = knn_indices[i_k_n, :next_K]
            if i in i_k_n_knn:
                NaNeighbor_map[i, i_k_n] = 1
                NaNeighbor_map[i_k_n, i] = 1
                NaN_distance[i, i_k_n] = knn_dis[i, next_K - 1]
                NaN_distance[i_k_n, i] = knn_dis[i, next_K - 1]

        zero_num = 0
        for i in range(row):
            if np.sum(NaNeighbor_map[i, :]) == 0:
                zero_num += 1

        if zero_num == 0:
            Flag = False
            break
        else:
            if zero_num == count_old:
                repeat_time += 1
                if repeat_time == 3:
                    Flag = False
                    break
            else:
                repeat_time = 0
                count_old = zero_num
                K = next_K

    return NaNeighbor_map, NaN_distance




