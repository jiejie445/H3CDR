# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
from model import GModel
from optimizer import Optimizer
from sklearn.model_selection import KFold
from sampler import Sampler
from GDSC.H3CDR.myutils import (roc_auc, evaluate_metrics, translate_result)
from extractDrugMolFeature import process_and_extract_features
from sklearn.decomposition import KernelPCA

#data_dir = dir_path(k=2) + "processed_data/"
data_dir = "E:\\pytorch_test\\H3CDR\\GDSC\\processed_data\\"

# 加载细胞系-药物关联矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_common_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
adj_mat_coo_data = sp.coo_matrix(cell_drug).data

# 加载药物-指纹特征矩阵
# drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
# feature_drug = np.array(drug_feature, dtype=np.float32)
# print("Shape of GDSC-drug_feature:", drug_feature.shape)

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

# 设置 Kernel PCA
kpca = KernelPCA(n_components=1000, kernel='rbf', gamma=131, random_state=42)
# 对每个模态的数据进行 Kernel PCA 降维
gene_data_reduced = kpca.fit_transform(gene)
cna_data_reduced = kpca.fit_transform(cna)
mutation_data_reduced = kpca.fit_transform(mutation)

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
        model = GModel(adj_mat=sampler.train_data, gene=gene_data_reduced, cna=cna_data_reduced,
                       mutation=mutation_data_reduced, drug=graph_feature_drug, n_filters=32, output_dim=128, device="cuda:0")
        opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                        evaluate_metrics, lr=1e-4, epochs=1000, device="cuda:0").to("cuda:0")
        epoch, true_data, predict_data = opt()
        epochs.append(epoch)
        true_datas = true_datas._append(translate_result(true_data))
        predict_datas = predict_datas._append(translate_result(predict_data))
file = open("./result_data/epochs.txt", "w")
file.write(str(epochs))
file.close()
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")

