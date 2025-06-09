# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
# from import_path import *
from CCLE.H3CDR.model import GModel
from CCLE.H3CDR.optimizer import Optimizer
from sklearn.model_selection import KFold
from CCLE.H3CDR.Entire_Drug_Cell.sampler import Sampler
from CCLE.H3CDR.myutils import roc_auc,evaluate_metrics, translate_result
from CCLE.H3CDR.extractDrugMolFeature import process_and_extract_features
# from H3CDR.Entire_Drug_Cell.Grid_algorithm import grid_main


# data_dir = dir_path(k=2) + "processed_data/"
data_dir = "E:\\pytorch_test\\H3CDR\\CCLE\\processed_data\\"

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_drug).data

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)
print("Shape of CCLE-drug_feature:", drug_feature.shape)

# 这里是加载药物的图级特征
# new_drug_feature = pd.read_csv(data_dir + "drug_graph_features.csv", index_col=0)
# graph_feature_drug = np.array(new_drug_feature, dtype=np.float32)

dataset_path = '../../processed_data/24drug_graph_feat'  # Path to your .hkl files
Graph_feature_drug = process_and_extract_features(dataset_path)
graph_feature_drug = Graph_feature_drug.to_numpy(dtype=np.float32)
print("Shape of GDSC-new_drug_feature:", graph_feature_drug.shape)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)
print("Shape of CCLE-gene:", gene.shape)
# 加载细胞系-cna特征矩阵
cna = pd.read_csv(data_dir + "cna_feature.csv", index_col=0, header=0)
cna = np.array(cna, dtype=np.float32)
print("Shape of CCLE-cna:", cna.shape)
# 加载细胞系-mutaion特征矩阵
mutation = pd.read_csv(data_dir + "mutation_feature.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)
print("Shape of CCLE-mutation:", mutation.shape)
# 加载null_mask
null_mask = np.zeros(cell_drug.shape, dtype=np.float32)

k = 5

kfold = KFold(n_splits=k, shuffle=True, random_state=11)

epochs = []
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

n_kfold = 5
for fold in range(n_kfold):
    for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0])):
        sampler = Sampler(cell_drug, train_index, test_index, null_mask)
        # model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=2, iterates=3,
        #                feature_drug=feature_drug, n_hid1=192, n_hid2=64, alpha=8.70, device="cuda:0")
        model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, null_mask=null_mask, sigma=2, k=11, iterates=3,
                       feature_drug=feature_drug, graph_feature_drug=graph_feature_drug, n_hid1=460, n_hid2=384,
                       dim=460, alpha=8.70, device="cuda:0")  # 这里把嵌入特征的维度变更为2倍。 n_heads=4,n_layers=2, dropout=0.5,
        opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                        evaluate_metrics, lr=1e-3, epochs=1000, device="cuda:0").to("cuda:0")
        epoch, true_data, predict_data = opt()

        epochs.append(epoch)
        true_datas = true_datas._append(translate_result(true_data))
        predict_datas = predict_datas._append(translate_result(predict_data))
file = open("./result_data/epochs.txt", "w")
file.write(str(epochs))
file.close()
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")


"""
# 网格计算超参数，除alpha外
save_format = "{:^5d}{:^5d}{:^5d}{:^7d}{:^7d}{:7.2f}{:^9.5f}{:^9.4f}"
file = open("grid_result.txt", "w")

sigmas = [2, 3, 5, 7, 9]
knns = [2, 3, 5, 7, 9, 11]
iterates = [2, 3, 5, 7, 9]
n_hid1s = [36, 64, 128, 192]
n_hid2s = [36, 64, 128, 192]
lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

for sigma in sigmas:
    for knn in knns:
        for iterate in iterates:
            for n_hid1 in n_hid1s:
                for n_hid2 in n_hid2s:
                    for lr in lrs:
                        grid_main(fold_k=k, random_state=11, original_adj_mat=cell_drug, null_mask=null_mask,
                                  gene=gene, cna=cna, mutation=mutation, drug_feature=feature_drug, sigma=sigma,
                                  knn=knn, iterates=iterate, n_hid1=n_hid1, n_hid2=n_hid2, alpha=5.74,
                                  evaluate_fun=roc_auc, lr=lr, epochs=1000, device="cuda", str_format=save_format,
                                  file=file)
file.close()
"""

"""
file = open("alpha_grid_result.txt", "w")
save_format = "{:^5d}{:^5d}{:^5d}{:^7d}{:^7d}{:7.2f}{:^9.5f}{:^9.4f}"
alphas = np.linspace(8, 9, 101)
for alpha in alphas:
    print("alpha:", alpha)
    grid_main(fold_k=k, random_state=11, original_adj_mat=cell_drug, null_mask=null_mask,
              gene=gene, cna=cna, mutation=mutation, drug_feature=feature_drug, sigma=2,
              knn=2, iterates=3, n_hid1=192, n_hid2=64, alpha=alpha,
              evaluate_fun=roc_auc, lr=1e-3, epochs=1000, device="cuda", str_format=save_format,
              file=file)
file.close()
"""
