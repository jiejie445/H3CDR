# coding: utf-8
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
#from import_path import *
from GDSC.H3CDR.model import GModel
from GDSC.H3CDR.optimizer import Optimizer
from sklearn.model_selection import KFold
from sampler import Sampler
from GDSC.H3CDR.myutils import (roc_auc, evaluate_metrics, translate_result)
from GDSC.H3CDR.extractDrugMolFeature import process_and_extract_features
# from H3CDR.Entire_Drug_Cell.Grid_algorithm import grid_main


#data_dir = dir_path(k=2) + "processed_data/"
data_dir = "E:\\pytorch_test\\MOFGCN-main\\GDSC\\processed_data\\"

# 加载细胞系-药物关联矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_common_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_drug).data

# 加载药物-指纹特征矩阵
drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
feature_drug = np.array(drug_feature, dtype=np.float32)
print("Shape of GDSC-drug_feature:", drug_feature.shape)

# 这里是加载药物的图级特征
# new_drug_feature = pd.read_csv(data_dir + "drug_graph_features.csv", index_col=0)
# graph_feature_drug = np.array(new_drug_feature, dtype=np.float32)

dataset_path = '../../processed_data/228drug_graph_feat'  # Path to your .hkl files
Graph_feature_drug = process_and_extract_features(dataset_path)
graph_feature_drug = Graph_feature_drug.to_numpy(dtype=np.float32)
print("Shape of GDSC-new_drug_feature:", graph_feature_drug.shape)

# 加载细胞系-基因特征矩阵
gene = pd.read_csv(data_dir + "cell_gene_feature.csv", index_col=0, header=0)
gene = np.array(gene, dtype=np.float32)
print("Shape of GDSC-gene:", gene.shape)

# 加载细胞系-cna特征矩阵
cna = pd.read_csv(data_dir + "cell_gene_cna.csv", index_col=0, header=0)
cna = cna.fillna(0)
cna = np.array(cna, dtype=np.float32)
print("Shape of GDSC-cna:", cna.shape)

# 加载细胞系-mutaion特征矩阵
mutation = pd.read_csv(data_dir + "cell_gene_mutation.csv", index_col=0, header=0)
mutation = np.array(mutation, dtype=np.float32)
print("Shape of GDSC-mutation:", mutation.shape)

# 加载null_mask
null_mask = pd.read_csv(data_dir + "null_mask.csv", index_col=0, header=0)
null_mask = np.array(null_mask, dtype=np.float32)


epochs = []
true_datas = pd.DataFrame()

predict_datas = pd.DataFrame()
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=11)

n_kfolds = 5
for n_kfold in range(n_kfolds):
    for train_index, test_index in kfold.split(np.arange(adj_mat_coo_data.shape[0])):
        sampler = Sampler(cell_drug, train_index, test_index, null_mask)
        # model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, sigma=2, k=11, iterates=3,
        #                feature_drug=feature_drug, n_hid1=1190, n_hid2=384,
        #                dim=1190, alpha=5.74,  device="cuda:0")   #这里把嵌入特征的维度变更为2倍。 n_heads=4,n_layers=2, dropout=0.5,
        model = GModel(adj_mat=sampler.train_data, gene=gene, cna=cna, mutation=mutation, null_mask=null_mask, sigma=2, k=11, iterates=3,
                       feature_drug=feature_drug, graph_feature_drug=graph_feature_drug, n_hid1=1190, n_hid2=384,
                       dim=1190, alpha=5.74, device="cuda:0")  # 这里把嵌入特征的维度变更为2倍。 n_heads=4,n_layers=2, dropout=0.5,

        opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                        evaluate_metrics, lr=5e-4, epochs=1000, device="cuda:0").to("cuda:0")
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
alphas = np.linspace(start=4.5, stop=6.5, num=101)
save_format = "{:^10.5f}{:^10.4f}"
save_file = open("./alpha_grid_result.txt", "w")
for alpha in alphas:
    alpha = float(alpha)
    grid_main(fold_k=5, random_state=11, original_adj_mat=cell_drug, null_mask=null_mask, gene=gene,
              cna=cna, mutation=mutation, drug_feature=feature_drug, sigma=2, knn=11,
              iterates=3, n_hid1=192, n_hid2=36, alpha=alpha, evaluate_fun=roc_auc, lr=5e-4,
              epochs=1000, neg_sample_times=9, device="cuda", str_format=save_format,
              file=save_file)
save_file.close()
"""
