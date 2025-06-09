# coding: utf-8
import numpy as np
import pandas as pd
from GDSC.H3CDR.myutils import roc_auc, translate_result
from GDSC.H3CDR.New_Drug_Cell.MOFGCN_New_target import mofgcn_new_target
from GDSC.H3CDR.extractDrugMolFeature import process_and_extract_features

# data_dir = dir_path(k=2) + "processed_data/"
data_dir = "E:\\pytorch_test\\H3CDR\\GDSC\\processed_data\\"

# 加载细胞系-药物矩阵
cell_drug = pd.read_csv(data_dir + "cell_drug_common_binary.csv", index_col=0, header=0)
cell_drug = np.array(cell_drug, dtype=np.float32)
cell_sum = np.sum(cell_drug, axis=1)
drug_sum = np.sum(cell_drug, axis=0)
print("dasatini:",drug_sum[57])   #下标57，第58列
print("第51列",drug_sum[50])   #下标50，第51列
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

#
# target_dim = [0, 1]
# n_kfold = 20
# file_drug = open("./result_data/new_drug_result.txt", "w")
# file_cell = open("./result_data/new_cell_result.txt", "w")
# for dim in target_dim:
#     for target_index in np.arange(cell_drug.shape[dim]):
#         if dim:
#             if drug_sum[target_index] < 10:
#                 continue
#         else:
#             if cell_sum[target_index] < 10:
#                 continue
#         epochs = []
#         true_data_s = pd.DataFrame()
#         predict_data_s = pd.DataFrame()
#         for fold in range(n_kfold):
#             epoch, true_data, predict_data = mofgcn_new_target(gene=gene, cna=cna, mutation=mutation,
#                                                                drug_feature=feature_drug, drug_graph_feature=graph_feature_drug,
#                                                                response_mat=cell_drug,
#                                                                null_mask=null_mask, target_dim=dim,
#                                                                target_index=target_index, evaluate_fun=roc_auc,
#                                                                device="cuda:0")
#             true_data_s = true_data_s._append(translate_result(true_data))
#             predict_data_s = predict_data_s._append(translate_result(predict_data))
#             epochs.append(epoch)
#         if dim:
#             file_drug.write(str(target_index) + ":" + str(epochs) + "\n")
#             true_data_s.to_csv("./result_data/drug_" + str(target_index) + "_true_data.csv")
#             predict_data_s.to_csv("./result_data/drug_" + str(target_index) + "_predict_data.csv")
#         else:
#             file_cell.write(str(target_index) + ":" + str(epochs) + "\n")
#             true_data_s.to_csv("./result_data/cell_" + str(target_index) + "_true_data.csv")
#             predict_data_s.to_csv("./result_data/cell_" + str(target_index) + "_predict_data.csv")



target_dim = [0, 1]
n_kfold = 40
file_drug = open("./result_data/new_drug_result.txt", "w")
file_cell = open("./result_data/new_cell_result.txt", "w")
dim = 1
for target_index in np.arange(cell_drug.shape[dim]):
    if dim:
        if target_index != 50:
            continue
    else:
        if cell_sum[target_index] < 10:
            continue
    epochs = []
    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()

    for fold in range(n_kfold):
        epoch, true_data, predict_data = mofgcn_new_target(gene=gene, cna=cna, mutation=mutation,
                                                           drug_feature=feature_drug, drug_graph_feature=graph_feature_drug,
                                                           response_mat=cell_drug,
                                                           null_mask=null_mask, target_dim=dim,
                                                           target_index=target_index, evaluate_fun=roc_auc,
                                                           device="cuda:0")
        true_data_s = true_data_s._append(translate_result(true_data))
        predict_data_s = predict_data_s._append(translate_result(predict_data))
        epochs.append(epoch)
    if dim:
        file_drug.write(str(target_index) + ":" + str(epochs) + "\n")
        true_data_s.to_csv("./result_data/drug_" + str(target_index) + "_true_data.csv")
        predict_data_s.to_csv("./result_data/drug_" + str(target_index) + "_predict_data.csv")
    else:
        file_cell.write(str(target_index) + ":" + str(epochs) + "\n")
        true_data_s.to_csv("./result_data/cell_" + str(target_index) + "_true_data.csv")
        predict_data_s.to_csv("./result_data/cell_" + str(target_index) + "_predict_data.csv")

file_drug.close()
file_cell.close()
