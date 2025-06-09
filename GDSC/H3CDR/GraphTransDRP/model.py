import torch
import numpy as np
from abc import ABC
import torch.nn as nn
import torch.nn.functional as fun
from myutils import  full_kernel, sparse_kernel, torch_corr_x_y, \
    scale_sigmoid_activation_function, scale_sigmoid
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class FusionFeature(nn.Module, ABC):
    def __init__(self, gene, cna, mutation, n_filters=32, output_dim=128, device="cuda:0"):
        super(FusionFeature, self).__init__()
        # cell line gene feature
        self.gene = torch.from_numpy(gene).to(device)
        self.cna = torch.from_numpy(cna).to(device)
        self.mutation = torch.from_numpy(mutation).to(device)
        self.conv_xt_ge_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8).to(device)
        self.pool_xt_ge_1 = nn.MaxPool1d(3)
        self.conv_xt_ge_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8).to(device)
        self.pool_xt_ge_2 = nn.MaxPool1d(3)
        self.conv_xt_ge_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8).to(device)
        self.pool_xt_ge_3 = nn.MaxPool1d(3)
        self.fc1_xt_gene = nn.Linear(4096, output_dim).to(device)
        self.device = device

        # cell line mut feature
        self.conv_xt_mut_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8).to(device)
        self.pool_xt_mut_1 = nn.MaxPool1d(3)
        self.conv_xt_mut_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8).to(device)
        self.pool_xt_mut_2 = nn.MaxPool1d(3)
        self.conv_xt_mut_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8).to(device)
        self.pool_xt_mut_3 = nn.MaxPool1d(3)
        self.fc1_xt_mut = nn.Linear(4096, output_dim).to(device)

        # cell line cna feature
        self.conv_xt_cna_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8).to(device)
        self.pool_xt_cna_1 = nn.MaxPool1d(3)
        self.conv_xt_cna_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8).to(device)
        self.pool_xt_cna_2 = nn.MaxPool1d(3)
        self.conv_xt_cna_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8).to(device)
        self.pool_xt_cna_3 = nn.MaxPool1d(3)
        self.fc1_xt_cna = nn.Linear(4096, output_dim).to(device)

    def forward(self):
        # gene 的形状是 (batch_size, input_dim, seq_length)，其中 seq_length = 19450
        gene = self.gene[:, None, :]  # 在第二维增加一个通道维度 (batch_size, 1, seq_length)
        gene = self.conv_xt_ge_1(gene)
        gene = fun.relu(gene)
        gene = self.pool_xt_ge_1(gene)
        gene = self.conv_xt_ge_2(gene)
        gene = fun.relu(gene)
        gene = self.pool_xt_ge_2(gene)
        gene = self.conv_xt_ge_3(gene)
        gene = fun.relu(gene)
        gene = self.pool_xt_ge_3(gene)
        gene = gene.view(gene.size(0), -1)  # 展平
        # gene = gene.flatten(start_dim=1)  # 展平
        gene = self.fc1_xt_gene(gene)  # 全连接层


        # mutation input feed-forward:
        mut = self.mutation
        mut = mut[:, None, :]
        mut = self.conv_xt_mut_1(mut)
        mut = fun.relu(mut)
        mut = self.pool_xt_mut_1(mut)
        mut = self.conv_xt_mut_2(mut)
        mut = fun.relu(mut)
        mut = self.pool_xt_mut_2(mut)
        mut = self.conv_xt_mut_3(mut)
        mut = fun.relu(mut)
        mut = self.pool_xt_mut_3(mut)
        xt_mut = mut.view(-1, mut.shape[1] * mut.shape[2])
        # xt_mut = mut.flatten(start_dim=1)  # Flatten for feeding into fc
        xt_mut = self.fc1_xt_mut(xt_mut)

        # cna input feed-forward:
        cna = self.cna
        cna = cna[:, None, :]
        cna = self.conv_xt_cna_1(cna)
        cna = fun.relu(cna)
        cna = self.pool_xt_cna_1(cna)
        cna = self.conv_xt_cna_2(cna)
        cna = fun.relu(cna)
        cna = self.pool_xt_cna_2(cna)
        cna = self.conv_xt_cna_3(cna)
        cna = fun.relu(cna)
        cna = self.pool_xt_cna_3(cna)
        xt_cna = cna.view(-1, cna.shape[1] * cna.shape[2])
        # xt_cna = cna.flatten(start_dim=1)  # Flatten for feeding into fc
        xt_cna = self.fc1_xt_cna(xt_cna)
        feature = torch.cat([gene, xt_cna, xt_mut], dim=1)

        return gene

class GDecoder(nn.Module, ABC):
    def __init__(self):
        super(GDecoder, self).__init__()
        self.lm_cell = nn.Linear(384, 128, bias=False)
        self.lm_drug = nn.Linear(384, 128, bias=False)

    def forward(self, cell, drug):
        # cell = self.lm_cell(cell)
        drug = self.lm_drug(drug)
        # print("在decoder中cell的形状：",cell.shape)
        # print("在decoder中cell的形状：",drug.shape)
        output = torch_corr_x_y(cell, drug)
        output = scale_sigmoid(output, alpha=8)
        return output

class GModel(nn.Module, ABC):
    def __init__(self, adj_mat, gene, cna, mutation, drug, n_filters, output_dim, device):
        super(GModel, self).__init__()
        # 特征融合模块
        self.fusioner = FusionFeature(gene, cna, mutation, n_filters, output_dim, device)
        self.feature = self.fusioner()
        self.drug = torch.tensor(drug).to(device)
        self.decoder = GDecoder()
    def forward(self):
        outPut = self.decoder(self.feature, self.drug)
        return outPut

class Early(object):
    def __init__(self, tolerance: int, data_len: int):
        self.auc = np.zeros(tolerance, dtype=np.float32)
        self.epoch = np.zeros(tolerance, dtype=np.int_)
        self.predict_data = torch.zeros((tolerance, data_len), dtype=torch.float32)
        self.tolerance = tolerance
        self.len = 0

    def push_data(self, auc, epoch, predict_data):
        i = self.len % self.tolerance
        self.auc[i] = auc
        self.epoch[i] = epoch
        self.predict_data[i, :] = predict_data
        self.len = self.len + 1

    def average(self):
        if self.len < self.tolerance:
            avg = 0
        else:
            avg = np.mean(self.auc)
        return avg

class EarlyStop(object):
    def __init__(self, tolerance: int, data_len: int):
        self.early = Early(tolerance=tolerance, data_len=data_len)
        self.auc_pre = None
        self.epoch_pre = None
        self.predict_data_pre = None

    def stop(self, auc, epoch, predict_data):
        avg_pre = self.early.average()
        self.auc_pre = self.early.auc.copy()
        self.epoch_pre = self.early.epoch.copy()
        self.predict_data_pre = self.early.predict_data.clone()
        self.early.push_data(auc=auc, epoch=epoch, predict_data=predict_data)
        avg_next = self.early.average()
        flag = False
        if avg_pre > avg_next:
            flag = True
        return flag

    def get_best_index(self):
        best_index = np.argmax(self.auc_pre)
        if self.epoch_pre[best_index] == 0:
            self.auc_pre[best_index] = 0
            best_index = np.argmax(self.auc_pre)
        return best_index
