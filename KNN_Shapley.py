import scipy.special
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN


def KNN_LOO(X_ori_aug, y_ori_aug, X_test, y_test, K, imbalance):
    N = X_ori_aug.shape[0]
    loos = np.zeros(N)
    for i in range(N):
        knn_with_instance = KNN(n_neighbors=K, weights='uniform')
        knn_with_instance.fit(X_ori_aug, y_ori_aug)
        preds1 = knn_with_instance.predict(X_test)
        if imbalance:
            acc1 = roc_auc_score(y_test, preds1)
        else:
            acc1 = accuracy_score(y_test, preds1)

        knn_without_instance = KNN(n_neighbors=K, weights='uniform')
        X_ori_aug_i = np.delete(X_ori_aug, i, axis=0)
        y_ori_aug_i = np.delete(y_ori_aug, i, axis=0)
        knn_without_instance.fit(X_ori_aug_i, y_ori_aug_i)
        preds2 = knn_without_instance.predict(X_test)
        if imbalance:
            acc2 = roc_auc_score(y_test, preds2)
        else:
            acc2 = accuracy_score(y_test, preds2)
        
        loo_value = acc1 - acc2
        loos[i] = loo_value

    loos = np.array(loos)
    return loos



def get_true_KNN(x_trn, x_tst):
    N = x_trn.shape[0]
    N_tst = x_tst.shape[0]
    x_tst_knn_gt = np.zeros((N_tst, N))
    for i_tst in tqdm(range(N_tst)):
        dist_gt = np.zeros(N)
        for i_trn in range(N):
            dist_gt[i_trn] = np.linalg.norm(x_trn[i_trn, :] - x_tst[i_tst, :], 2)
        x_tst_knn_gt[i_tst, :] = np.argsort(dist_gt)
    return x_tst_knn_gt.astype(int)



def compute_betas(x_tst_knn_gt_j_UV, ind_arr_UV):
    betas = []
    ind_arr_j_UV = ind_arr_UV[x_tst_knn_gt_j_UV]
    for i in range(len(x_tst_knn_gt_j_UV)):
        if ind_arr_j_UV[i] == 0:
            betas.append(int(ind_arr_j_UV[:i].sum()))
    return np.asarray(betas)



def compute_KNN_shapley(y_trn, x_tst_knn_gt, y_tst, K):
    N = y_trn.shape[0]
    N_tst = x_tst_knn_gt.shape[0]
    knn_shap_vals = np.zeros((N_tst, N))
    for j in tqdm(range(N_tst)):
        knn_shap_vals[j, x_tst_knn_gt[j, -1]] = (y_trn[x_tst_knn_gt[j, -1]] == y_tst[j]) / N
        for i in np.arange(N - 2, -1, -1):
            knn_shap_vals[j, x_tst_knn_gt[j, i]] = knn_shap_vals[j, x_tst_knn_gt[j, i + 1]] + \
            (int(y_trn[x_tst_knn_gt[j, i]] == y_tst[j]) - int(y_trn[x_tst_knn_gt[j, i + 1]] == y_tst[j]))\
                                                   / K * min([K, i + 1]) / (i + 1)
    return knn_shap_vals.mean(axis=0)



def compute_KNN_Asymmetric_Shapley(y_trn, x_tst_knn_gt, y_tst, K, s_class_dic):
    N_tst = y_tst.shape[0]
    knn_asym_shap_vals = {}

    for i_class in sorted(s_class_dic.keys()):
        U = []
        for j_class in range(i_class):
            U.extend(s_class_dic[j_class])
        V = s_class_dic[i_class]
        N_U, N_V = len(U), len(V)
        ind_arr_UV = np.zeros(len(U)+len(V)).astype(int)
        ind_arr_UV[:len(U)] = 1
        knn_asym_shap_i_class = np.zeros((N_tst, N_V))

        for j_tst in tqdm(range(N_tst)):
            x_tst_knn_gt_j = x_tst_knn_gt[j_tst, :].flatten()
            x_tst_knn_gt_j_UV = [i for i in x_tst_knn_gt_j if i in U or i in V]
            x_tst_knn_gt_j_U = [i for i in x_tst_knn_gt_j if i in U]
            x_tst_knn_gt_j_V = [i for i in x_tst_knn_gt_j if i in V]
            
            betas = compute_betas(x_tst_knn_gt_j_UV, ind_arr_UV)

            if K - betas[-1] <= 0:
                knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[-1]-len(U)] = 0
            elif K - betas[-1] > 0:
                B = 0
                if N_U > betas[-1]:
                    for k in range(0, K - betas[-1]):
                            B += int(y_trn[x_tst_knn_gt_j_U[K - k - 1]]==y_tst[j_tst])/K/N_V
                knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[-1]-len(U)] = int(y_trn[x_tst_knn_gt_j_V[-1]]==y_tst[j_tst])*(K - betas[-1])/K/N_V - B

            for l in np.arange(N_V-2, -1, -1):
                if K - betas[l] <= 0:
                    knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[l]-len(U)] = knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[l+1] - len(U)]

                elif K - betas[l] > 0 and K - betas[l + 1] > 0:
                    A = 0
                    if betas[l] < betas[l + 1]:
                        for k in range(K - betas[l+1], N_V - 1):
                            for m in range(K - betas[l+1], min([K-betas[l]-1, k]) + 1):     
                                A += scipy.special.binom(l, m) * scipy.special.binom(N_V - l - 2, k - m) * \
                                (int(y_trn[x_tst_knn_gt_j_V[l]] == y_tst[j_tst]) -
                                int(y_trn[x_tst_knn_gt_j_U[K - m - 1]] == y_tst[j_tst]))/K/scipy.special.binom(N_V - 2, k)
                        A = A / (N_V - 1)
                    knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[l] - len(U)] = knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[l + 1]- len(U)] + A + \
                                                    (int(y_trn[x_tst_knn_gt_j_V[l]] ==y_tst[j_tst])-
                                                     int(y_trn[x_tst_knn_gt_j_V[l + 1]] ==y_tst[j_tst])) / K * min([K - betas[l + 1], l + 1]) / (l + 1)

                elif K - betas[l] > 0 and K - betas[l + 1] <= 0:
                    A = 0
                    for k in range(0, N_V - 1):
                        for m in range(0, min([K - betas[l] - 1, k]) + 1):
                            A += scipy.special.binom(l, m) * scipy.special.binom(N_V - l - 2, k - m) * \
                                     (int(y_trn[x_tst_knn_gt_j_V[l]] == y_tst[j_tst]) -
                                      int(y_trn[x_tst_knn_gt_j_U[K - m - 1]] == y_tst[j_tst])) / K / scipy.special.binom(N_V - 2, k)
                    A = A / (N_V - 1)
                    knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[l] - len(U)] = knn_asym_shap_i_class[j_tst, x_tst_knn_gt_j_V[l + 1] - len(U)] + A

            knn_asym_shap_vals[i_class] = knn_asym_shap_i_class.mean(axis=0)


    knn_asym_shap_vals_final = np.zeros((0,))
    for i_class in sorted(knn_asym_shap_vals.keys()):
        knn_asym_shap_vals_final = np.append(knn_asym_shap_vals_final, knn_asym_shap_vals[i_class])

    return knn_asym_shap_vals_final



def get_VN(x_trn, x_tst, y_trn, y_tst, K):
    N = x_trn.shape[0]
    N_tst = x_tst.shape[0]
    x_tst_knn_gt = np.zeros((N_tst, N))
    for i_tst in tqdm(range(N_tst)):
        dist_gt = np.zeros(N)
        for i_trn in range(N):
            dist_gt[i_trn] = np.linalg.norm(x_trn[i_trn, :] - x_tst[i_tst, :], 2)
        x_tst_knn_gt[i_tst, :] = np.argsort(dist_gt) # 默认为从小到大的对应的索引
    x_tst_knn_gt = x_tst_knn_gt.astype(int)

    vn = 0
    for i_tst in range(N_tst):
        ids = x_tst_knn_gt[i_tst,:K]
        vn += (y_trn[ids] == y_tst[i_tst]).sum() / K
    return vn / len(y_tst)
