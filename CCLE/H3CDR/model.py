import torch
import numpy as np
from abc import ABC
import torch.nn as nn
import torch.nn.functional as fun
from myutils import exp_similarity, full_kernel, sparse_kernel, jaccard_coef, torch_corr_x_y, \
    scale_sigmoid_activation_function, scale_sigmoid


class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, nullMask, device="cpu"):
        super(ConstructAdjMatrix, self).__init__()
        self.adj_mat = original_adj_mat.to(device)
        self.null_mask = torch.tensor(nullMask, dtype=torch.float32).to(device)  # 标记哪些是未知的药物反应
        self.device = device

    def forward(self):
        n_cell = self.adj_mat.shape[0]
        n_drug = self.adj_mat.shape[1]
        cell_identity = torch.diag(torch.diag(torch.ones(n_cell, n_cell, dtype=torch.float, device=self.device)))
        drug_identity = torch.diag(torch.diag(torch.ones(n_drug, n_drug, dtype=torch.float, device=self.device)))
        cell_drug = torch.cat((cell_identity, self.adj_mat), dim=1)
        drug_cell = torch.cat((torch.t(self.adj_mat), drug_identity), dim=1)
        adj_matrix = torch.cat((cell_drug, drug_cell), dim=0)
        d = torch.diag(torch.pow(torch.sum(adj_matrix, dim=1), -1/2))
        identity = torch.diag(torch.diag(torch.ones(d.shape, dtype=torch.float, device=self.device)))
        # adj_matrix_hat = torch.add(identity, torch.mm(d, torch.mm(adj_matrix, d)))
        adj_matrix_hat = torch.mm(d, torch.mm(adj_matrix, d))
        adj_matrix_hat = torch.add(identity, adj_matrix_hat)

        # GUN网络中的邻接矩阵-对A做复杂二次幂运算
        U_adj = torch.mm(torch.mm(adj_matrix, adj_matrix), adj_matrix)
        resistMask = torch.add(self.adj_mat, self.null_mask)  # 在该矩阵中，1表示敏感或者未作实验，0表示耐药
        cell_drug_null = torch.cat((cell_identity, resistMask), dim=1)
        drug_cell_null = torch.cat((torch.t(resistMask), drug_identity), dim=1)
        adj_matrix_null = torch.cat((cell_drug_null, drug_cell_null), dim=0)
        U_adj_matrix = torch.mul(U_adj, adj_matrix_null)  # 将A*A中 耐药的置为0，敏感或未知的置为1。
        U_adj_matrix = (U_adj_matrix != 0).float()


        return adj_matrix_hat, U_adj_matrix


class ConstructCellAdjMatrix(nn.Module,ABC):
    def __init__(self, original_adj_mat, device="gpu" ):
        super(ConstructCellAdjMatrix,self).__init__()
        self.cellAdjMatrix = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        cellAdjMatrix = torch.mm(self.cellAdjMatrix, torch.t(self.cellAdjMatrix))
        num = cellAdjMatrix.shape[0]
        identity = torch.eye(num, dtype=torch.float, device=self.device)
        # cellAdjMatrix = torch.add(identity, cellAdjMatrix)
        cellAdjMatrix = torch.add(identity, cellAdjMatrix)
        d = torch.diag(torch.pow(torch.sum(cellAdjMatrix, dim=1), -1 / 2))
        # Step 2: 计算度矩阵并加入防止除零的小偏置
        # degree_sum = torch.sum(cellAdjMatrix, dim=1)
        # d = torch.diag(torch.pow(degree_sum + 1e-10, -1 / 2))  # 加入1e-10防止除零
        cellAdjMatrix = torch.add(identity,cellAdjMatrix)
        # cellAdjMatrix = torch.add(identity,cellAdjMatrix)
        result = torch.add(identity, torch.mm(d, torch.mm(cellAdjMatrix, d)))
        return result          #加1-求和-加1-加1

class ConstructDrugAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="gpu"):
        super(ConstructDrugAdjMatrix, self).__init__()
        self.drugAdjMatrix = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        drugAdjMatrix = torch.mm(torch.t(self.drugAdjMatrix), self.drugAdjMatrix)
        num = drugAdjMatrix.shape[0]
        identity = torch.eye(num, dtype=torch.float, device=self.device)
        # drugAdjMatrix = torch.add(identity, drugAdjMatrix)
        drugAdjMatrix = torch.add(identity, drugAdjMatrix)
        d = torch.diag(torch.pow(torch.sum(drugAdjMatrix, dim=1), -1 / 2))
        drugAdjMatrix = torch.add(identity, drugAdjMatrix)
        # drugAdjMatrix = torch.add(identity, drugAdjMatrix)
        result = torch.add(identity, torch.mm(d, torch.mm(drugAdjMatrix, d)))
        return result        #加1-求和-加1-加1

#U型结构里边使用到的邻接矩阵
class ConstructUNetAdjMatrix(nn.Module,ABC):
    def __init__(self, original_adj_mat, device="gpu" ):
        super(ConstructUNetAdjMatrix,self).__init__()
        self.UNetAdjMatrix = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        UNetAdjMatrix = torch.mm(torch.mm(self.UNetAdjMatrix, torch.t(self.UNetAdjMatrix)), self.UNetAdjMatrix)
        UNetAdjMatrix[UNetAdjMatrix != 0] = 1
        n_cell = self.UNetAdjMatrix.shape[0]
        n_drug = self.UNetAdjMatrix.shape[1]
        cell_identity = torch.eye(n_cell, dtype=torch.float, device=self.device)
        drug_identity = torch.eye(n_drug, dtype=torch.float, device=self.device)
        cell_drug = torch.cat((cell_identity, UNetAdjMatrix), dim=1)
        drug_cell = torch.cat((torch.t(UNetAdjMatrix), drug_identity), dim=1)
        poweredAdjMatrix = torch.cat((cell_drug, drug_cell), dim=0)

        return poweredAdjMatrix

class concatCellAndDrug(nn.Module, ABC):
    def __init__(self, cell_adj_mat, drug_adj_mat, org_adj, device="gpu"):
        super(concatCellAndDrug, self).__init__()
        self.cell = cell_adj_mat.to(device)
        self.drug = drug_adj_mat.to(device)
        self.org = org_adj.to(device)
        self.device = device

    def forward(self):
        cell_drug = torch.cat((self.cell, self.org), dim=1)
        drug_cell = torch.cat((torch.t(self.org), self.drug), dim=1)
        adj_matrix = torch.cat((cell_drug, drug_cell), dim=0)
        identity = torch.eye(460, dtype=torch.float, device=self.device)
        adj_matrix = torch.add(identity, adj_matrix)
        return adj_matrix

class OnlySimilarity(nn.Module, ABC):
    def __init__(self, gene, cna, mutation, sigma, k, iterates, feature_drug, graph_feature_drug, device="gpu"):
        super(OnlySimilarity, self).__init__()
        gene = torch.from_numpy(gene).to(device)
        cna = torch.from_numpy(cna).to(device)
        mutation = torch.from_numpy(mutation).to(device)
        sigma = torch.tensor(sigma, dtype=torch.float, device=device)
        feature_drug = torch.from_numpy(feature_drug).to(device)
        graph_feature_drug = torch.from_numpy(graph_feature_drug).to(device)

        self.gene_exp_similarity = exp_similarity(gene, sigma)
        self.cna_exp_similarity = exp_similarity(cna, sigma, normalize=False)  ##删除了这里的 normalize = False
        self.mutation_exp_similarity = exp_similarity(mutation, sigma, normalize=False)  ##删除了这里的 normalize = False
        self.drug_jac_similarity = jaccard_coef(feature_drug)
        self.drug_graph_similarity = graph_feature_drug

        self.k = k
        self.iterates = iterates
        self.device = device

    def fusion_cell_feature(self):
        gene_p = full_kernel(self.gene_exp_similarity)
        gene_s = sparse_kernel(self.gene_exp_similarity, k=self.k)
        cna_p = full_kernel(self.cna_exp_similarity)
        cna_s = sparse_kernel(self.cna_exp_similarity, k=self.k)
        mutation_p = full_kernel(self.mutation_exp_similarity)
        mutation_s = sparse_kernel(self.mutation_exp_similarity, k=self.k)
        two = torch.tensor(2, dtype=torch.float32, device=self.device)
        three = torch.tensor(3, dtype=torch.float32, device=self.device)
        it = 0
        while it < self.iterates:
            gene_p_next = torch.mm(torch.mm(gene_s, torch.div(torch.add(cna_p, mutation_p), two)), gene_s.t())
            cna_p_next = torch.mm(torch.mm(cna_s, torch.div(torch.add(gene_p, mutation_p), two)), cna_s.t())
            mutation_p_next = torch.mm(torch.mm(mutation_s, torch.div(torch.add(cna_p, gene_p), two)), mutation_s.t())
            gene_p = gene_p_next
            cna_p = cna_p_next
            mutation_p = mutation_p_next
            it += 1
        fusion_feature = torch.div(torch.add(torch.add(gene_p, cna_p), mutation_p), three)
        fusion_feature = fusion_feature.to(dtype=torch.float32)
        return fusion_feature

    def fusion_drug_feature(self):
        jac_p = full_kernel(self.drug_jac_similarity)
        jac_s = sparse_kernel(self.drug_jac_similarity,k=self.k)
        molGraph_p = full_kernel(self.drug_graph_similarity)
        molGraph_s = sparse_kernel(self.drug_graph_similarity,k=self.k)
        two = torch.tensor(2, dtype=torch.float32, device=self.device)
        it = 0
        while it < self.iterates:
            jac_p_next = torch.mm(torch.mm(jac_s, molGraph_p), jac_s.t())
            molGraph_p_next = torch.mm(torch.mm(molGraph_s, jac_p), molGraph_s.t())
            jac_p = jac_p_next
            molGraph_p = molGraph_p_next
            it+=1
        fusion_drug_feature = torch.div(torch.add(jac_p,molGraph_p), two)
        fusion_drug_feature = fusion_drug_feature.to(dtype=torch.float32)
        return fusion_drug_feature

    def forward(self):
        drug_graph_feature = self.drug_graph_similarity
        # drug_graph_feature = full_kernel(drug_graph_feature)          # 数据层面的消融实验--去掉子结构指纹数据，使用药物分子图特征作为同构GCN的输入特征
        # drug_similarity = self.fusion_drug_feature()
        drug_similarity = full_kernel(self.drug_jac_similarity)
        # drug_similarity = full_kernel(self.drug_graph_similarity)
        # drug_similarity = (full_kernel(self.drug_jac_similarity) + full_kernel(self.drug_graph_similarity))/2
        cell_similarity = self.fusion_cell_feature()
        return cell_similarity, drug_similarity, drug_graph_feature

class OnlyConcatFeature(nn.Module, ABC):
    def __init__(self, device="gpu"):
        super(OnlyConcatFeature, self).__init__()
        self.device = device

    def forward(self, cell, drug):
        zeros1 = torch.zeros(cell.shape[0], drug.shape[1], dtype=torch.float32,
                             device=self.device)
        zeros2 = torch.zeros(drug.shape[0], cell.shape[1], dtype=torch.float32,
                             device=self.device)
        cell_zeros = torch.cat((cell, zeros1), dim=1)
        zeros_drug = torch.cat((zeros2, drug), dim=1)
        fusion_feature = torch.cat((cell_zeros, zeros_drug), dim=0)
        return fusion_feature


class FusionFeature(nn.Module, ABC):
    def __init__(self, gene, cna, mutation, sigma, k, iterates, feature_drug, graph_feature_drug, device="cpu"):
        super(FusionFeature, self).__init__()
        gene = torch.from_numpy(gene).to(device)
        cna = torch.from_numpy(cna).to(device)
        mutation = torch.from_numpy(mutation).to(device)
        sigma = torch.tensor(sigma, dtype=torch.float, device=device)
        feature_drug = torch.from_numpy(feature_drug).to(device)
        graph_feature_drug = torch.from_numpy(graph_feature_drug).to(device)
        self.gene_exp_similarity = exp_similarity(gene, sigma)
        self.cna_exp_similarity = exp_similarity(cna, sigma, normalize=False)
        self.mutation_exp_similarity = exp_similarity(mutation, sigma, normalize=False)
        self.drug_jac_similarity = jaccard_coef(feature_drug)
        self.drug_graph_similarity = graph_feature_drug
        self.k = k
        self.iterates = iterates
        self.device = device

    def fusion_cell_feature(self):
        gene_p = full_kernel(self.gene_exp_similarity)
        gene_s = sparse_kernel(self.gene_exp_similarity, k=self.k)
        cna_p = full_kernel(self.cna_exp_similarity)
        cna_s = sparse_kernel(self.cna_exp_similarity, k=self.k)
        mutation_p = full_kernel(self.mutation_exp_similarity)
        mutation_s = sparse_kernel(self.mutation_exp_similarity, k=self.k)
        two = torch.tensor(2, dtype=torch.float32, device=self.device)
        three = torch.tensor(3, dtype=torch.float32, device=self.device)
        it = 0

        # 三种组学时的迭代策略
        while it < self.iterates:
            gene_p_next = torch.mm(torch.mm(gene_s, torch.div(torch.add(cna_p, mutation_p), two)), gene_s.t())
            cna_p_next = torch.mm(torch.mm(cna_s, torch.div(torch.add(gene_p, mutation_p), two)), cna_s.t())
            mutation_p_next = torch.mm(torch.mm(mutation_s, torch.div(torch.add(cna_p, gene_p), two)), mutation_s.t())
            gene_p = gene_p_next
            cna_p = cna_p_next
            mutation_p = mutation_p_next
            it += 1
        fusion_feature = torch.div(torch.add(torch.add(gene_p, cna_p), mutation_p), three)
        fusion_feature = fusion_feature.to(dtype=torch.float32)

        # 两种组学时的迭代策略---去掉基因表达数据
        # while it < self.iterates:
        #     cna_p_next = torch.mm(torch.mm(cna_s, mutation_p), cna_s.t())
        #     mutation_p_next = torch.mm(torch.mm(mutation_s, cna_p), mutation_s.t())
        #     cna_p = cna_p_next
        #     mutation_p = mutation_p_next
        #     it += 1
        # fusion_feature = torch.div(torch.add(cna_p, mutation_p), two)
        # fusion_feature = fusion_feature.to(dtype=torch.float32)

        # 两种组学时的迭代策略---去掉CNA数据
        # while it < self.iterates:
        #     gene_p_next = torch.mm(torch.mm(gene_s, mutation_p), gene_s.t())
        #     mutation_p_next = torch.mm(torch.mm(mutation_s, gene_p), mutation_s.t())
        #     gene_p = gene_p_next
        #     mutation_p = mutation_p_next
        #     it += 1
        # fusion_feature = torch.div(torch.add(gene_p, mutation_p), two)
        # fusion_feature = fusion_feature.to(dtype=torch.float32)

        # 两种组学时的迭代策略---去掉mutation数据
        # while it < self.iterates:
        #     gene_p_next = torch.mm(torch.mm(gene_s, cna_p), gene_s.t())
        #     cna_p_next = torch.mm(torch.mm(cna_s, gene_p), cna_s.t())
        #     gene_p = gene_p_next
        #     cna_p = cna_p_next
        #     it += 1
        # fusion_feature = torch.div(torch.add(gene_p, cna_p), two)
        # fusion_feature = fusion_feature.to(dtype=torch.float32)

        # 1种组学时的迭代策略---仅保留基因表达数据
        # while it < self.iterates:
        #     gene_p_next = torch.mm(torch.mm(gene_s, gene_p), gene_s.t())
        #     gene_p = gene_p_next
        #     it += 1
        # fusion_feature = gene_p
        # fusion_feature = fusion_feature.to(dtype=torch.float32)

        # 1种组学时的迭代策略---仅保留mutation数据
        # while it < self.iterates:
        #     mutation_p_next = torch.mm(torch.mm(mutation_s, mutation_p), mutation_s.t())
        #     mutation_p = mutation_p_next
        #     it += 1
        # fusion_feature = mutation_p
        # fusion_feature = fusion_feature.to(dtype=torch.float32)


        # 1种组学时的迭代策略---仅保留CNA数据
        # while it < self.iterates:
        #     cna_p_next = torch.mm(torch.mm(cna_s, cna_p), cna_s.t())
        #     cna_p = cna_p_next
        #     it += 1
        # fusion_feature = cna_p
        # fusion_feature = fusion_feature.to(dtype=torch.float32)

        return fusion_feature

    def fusion_drug_feature(self):
        jac_p = full_kernel(self.drug_jac_similarity)
        jac_s = sparse_kernel(self.drug_jac_similarity,k=self.k)
        molGraph_p = full_kernel(self.drug_graph_similarity)
        molGraph_s = sparse_kernel(self.drug_graph_similarity,k=self.k)
        two = torch.tensor(2, dtype=torch.float32, device=self.device)
        it = 0
        while it < self.iterates:
            jac_p_next = torch.mm(torch.mm(jac_s, molGraph_p), jac_s.t())
            molGraph_p_next = torch.mm(torch.mm(molGraph_s, jac_p), molGraph_s.t())
            jac_p = jac_p_next
            molGraph_p = molGraph_p_next
            it+=1
        fusion_drug_feature = torch.div(torch.add(jac_p,molGraph_p), two)
        fusion_drug_feature = fusion_drug_feature.to(dtype=torch.float32)
        return fusion_drug_feature

    def forward(self):
        drug_similarity = full_kernel(self.drug_jac_similarity)       # 原方法-对药物的子结构指纹进行jaccard相似性计算，进一步计算得到药物的全核特征相似性数据
        # drug_similarity = full_kernel(self.drug_graph_similarity)     # 消融实验-根据药物的分子图数据得到药物的全核特征
        cell_similarity = self.fusion_cell_feature()
        zeros1 = torch.zeros(cell_similarity.shape[0], drug_similarity.shape[1], dtype=torch.float32,
                             device=self.device)
        zeros2 = torch.zeros(drug_similarity.shape[0], cell_similarity.shape[1], dtype=torch.float32,
                             device=self.device)
        cell_zeros = torch.cat((cell_similarity, zeros1), dim=1)
        zeros_drug = torch.cat((zeros2, drug_similarity), dim=1)
        fusion_feature = torch.cat((cell_zeros, zeros_drug), dim=0)


        return fusion_feature


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p=0.0):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        # h = self.drop(h)
        # 对邻接矩阵进行归一化
        d = torch.diag(torch.pow(torch.sum(g, dim=1), -1 / 2))
        identity = torch.eye(d.shape[0], dtype=torch.float, device="cuda:0")
        # g = torch.add(identity, g)
        adj_matrix_hat = torch.mm(d, torch.mm(g, d))
        adj_matrix_hat = torch.add(identity, adj_matrix_hat)
        h = torch.matmul(adj_matrix_hat, h)  # g 是邻接矩阵

        # h = torch.matmul(g, h)  # g 是邻接矩阵
        h = self.proj(h)
        h = self.act(h)

        return h

# 定义池化层
class Pool(nn.Module):
    def __init__(self, k, in_dim, p=0.0):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)

# 图池化和标准化
def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)

    un_g = g.bool().float()
    # un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx

# 邻接矩阵归一化
def norm_g(g):
    degrees = torch.sum(g, 1)
    d = torch.diag(torch.pow(degrees, -1/2))
    identity = torch.diag(torch.diag(torch.ones(d.shape)))
    identity = identity.to('cuda')
    g = torch.mm(d, torch.mm(g, d))
    g = torch.add(identity, g)
    # g = g / degrees.unsqueeze(-1)
    return g

# 定义Unpool层
class Unpool(nn.Module):
    def __init__(self, device="gpu"):
        super(Unpool, self).__init__()
        self.device = device

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        new_h = new_h + pre_h  # 这里修改了点
        return g, new_h

# 定义GraphUNet
class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        self.lm = nn.Linear(in_dim, out_dim, bias=False)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool())

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h

        # 编码阶段
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            # print("经过GCN之后h的形状：",h.shape)       # [1190,1190]、[595,1190]
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        # 底部GCN
        h = self.bottom_gcn(g, h)
        # print("经过底层GCN之后h的形状：", h.shape)      # [297, 1190]
        # 解码阶段
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)

        # 加上原始特征
        # h = h.add(org_h)
        # print("经过UNet之后h的特征维度：",h.shape)      1190*1190
        hs.append(h)
        # h = self.lm(h)
        # print("经过线性变换之后h的特征维度：",h.shape)    # 1190*384
        return h  # 返回最后一层的特征

class GCNLayer(nn.Module, ABC):
    def __init__(self, in_dim, n_hid):
        super(GCNLayer, self).__init__()
        self.lm = nn.Linear(in_dim, n_hid, bias=False)

    def forward(self, adj_mat, feature):
        # if torch.isnan(adj_mat).any():
        #     print("Adjacency matrix contains NaN values")
        input_adj_mat = torch.mm(adj_mat, feature)
        lm_out = self.lm(input_adj_mat)
        lm_out = fun.leaky_relu(lm_out)
        return lm_out

class GEncoder(nn.Module, ABC):
    def __init__(self, adj_mat, feature, n_hid):
        super(GEncoder, self).__init__()
        self.adj_mat = adj_mat
        self.feature = feature
        self.lm = nn.Linear(feature.shape[1], n_hid, bias=False)

    def forward(self):
        input_adj_mat = torch.mm(self.adj_mat, self.feature)
        lm_out = self.lm(input_adj_mat)
        lm_out = fun.leaky_relu(lm_out)
        return lm_out

# class GDecoder(nn.Module, ABC):
#     def __init__(self, n_cell, n_drug, n_hid1, n_hid2, alpha):
#         super(GDecoder, self).__init__()
#         self.n_cell = n_cell
#         self.n_drug = n_drug
#         self.alpha = alpha
#         self.lm_cell = nn.Linear(n_hid1, n_hid2, bias=False)
#         self.lm_drug = nn.Linear(n_hid1, n_hid2, bias=False)
#
#     def forward(self, encode_output):
#         z_cell, z_drug = torch.split(encode_output, [self.n_cell, self.n_drug], dim=0)
#         cell = self.lm_cell(z_cell)
#         drug = self.lm_drug(z_drug)
#         output = torch_corr_x_y(cell, drug)
#         output = scale_sigmoid_activation_function(output, alpha=self.alpha)
#         return output


class GDecoder(nn.Module, ABC):
    def __init__(self, n_cell, n_drug, n_hid1, n_hid2, alpha, device):
        super(GDecoder, self).__init__()
        self.n_cell = n_cell
        self.n_drug = n_drug
        self.alpha = alpha

        self.lm_cell = nn.Linear(436, 72, bias=False)
        self.lm_drug = nn.Linear(24, 72, bias=False)

    # def forward(self, encode_output):
    #     z_cell = encode_output[:962, :962]
    #     z_drug = encode_output[962:, 962:]
    #     cell = self.lm_cell(z_cell)
    #     drug = self.lm_drug(z_drug)
    #     output = torch_corr_x_y(cell, drug)
    #     output = scale_sigmoid_activation_function(output, alpha=self.alpha)
    #     return output

    # def forward(self, encode_output, cellGCN, drugGCN):
    #     z_cell = encode_output[:962, :962]
    #     z_drug = encode_output[962:, 962:]
    #     # z_cell, z_drug = torch.split(encode_output, [self.n_cell, self.n_drug], dim=0)
    #     z_cell = torch.add(z_cell, cellGCN)
    #     z_drug = torch.add(z_drug, drugGCN)
    #     cell = self.lm_cell(z_cell)
    #     drug = self.lm_drug(z_drug)
    #     output = torch_corr_x_y(cell, drug)
    #     # output = torch_muIn(cell, drug)
    #     # output = scale_sigmoid_activation_function(output, alpha=self.alpha)
    #     output = scale_sigmoid(output, alpha=8)
    #     return output

    def forward(self, GUN_output, GCN_output, cellGCN, drugGCN):
        GUN_cell = GUN_output[:436, :436]
        GUN_drug = GUN_output[436:, 436:]
        # 这里是异构GCN的特征
        GCN_cell = GCN_output[:436, :436]
        GCN_drug = GCN_output[436:, 436:]
        # z_cell, z_drug = torch.split(encode_output, [self.n_cell, self.n_drug], dim=0)
        z_cell = torch.add(GUN_cell, GCN_cell)
        z_drug = torch.add(GUN_drug, GCN_drug)

        #额外再加上同构GCN的特征
        z_cell = torch.add(z_cell, cellGCN)
        z_drug = torch.add(z_drug, drugGCN)

        cell = self.lm_cell(z_cell)
        drug = self.lm_drug(z_drug)
        output = torch_corr_x_y(cell, drug)
        # output = torch_muIn(cell, drug)
        # output = scale_sigmoid_activation_function(output, alpha=self.alpha)
        output = scale_sigmoid(output, alpha=8)
        return output


class GModel(nn.Module, ABC):
    def __init__(self, adj_mat, gene, cna, mutation, null_mask, sigma, k, iterates, feature_drug, graph_feature_drug,  n_hid1,
                 n_hid2, dim, alpha, act=nn.LeakyReLU(), drop_p=0, ks=[0.5,0.5], device="gpu"):
        super(GModel, self).__init__()
        # 特征融合模块
        self.construct_adj_matrix = ConstructAdjMatrix(adj_mat, null_mask, device=device)
        self.construct_cell_adj_matrix = ConstructCellAdjMatrix(adj_mat,device)     #!!  计算细胞系的邻接矩阵二次幂
        self.construct_drug_adj_matrix = ConstructDrugAdjMatrix(adj_mat,device)     #!!  计算药物的邻接矩阵二次幂
        self.construct_Unet_adj_matrix = ConstructUNetAdjMatrix(adj_mat,device)     #计算在U型结构中使用的邻接矩阵

        self.fusioner = FusionFeature(gene, cna, mutation, sigma=sigma, k=k, iterates=iterates,
                                      feature_drug=feature_drug, graph_feature_drug=graph_feature_drug, device=device)
        self.similarFeature = OnlySimilarity(gene, cna, mutation,sigma, k=k, iterates=iterates,
                                      feature_drug=feature_drug, graph_feature_drug=graph_feature_drug, device=device)     #!!  计算细胞系和药物的相似性矩阵
        # 构造邻接矩阵
        self.adj_matrix_hat, self.Unet_adj_matrix = self.construct_adj_matrix()
        self.cell_adj_matrix = self.construct_cell_adj_matrix()     #!!   得到细胞系的二次邻接矩阵
        self.drug_adj_matrix = self.construct_drug_adj_matrix()     #!!   得到药物的二次邻接矩阵
        # self.Unet_adj_matrix = self.construct_Unet_adj_matrix()     # 得到在GUNet中使用的邻接矩阵
        # self.construct_concat_adj_matrix = concatCellAndDrug(self.cell_adj_matrix, self.drug_adj_matrix, adj_mat, device)     #!!
        # self.concat_adj_matrix = self.construct_concat_adj_matrix()     #!!      将二次邻接矩阵进行拼接

        self.feature = self.fusioner()
        self.cellFeature, self.drugFeature, self.drug_graph_feature = self.similarFeature()       #!!   得到细胞系和药物各自的相似性矩阵
        self.GCNCellFeature = GCNLayer(436, 436)     #!!     定义细胞系的GCNLayer结构
        self.GCNDrugFeature = GCNLayer(24, 24)     #!!     定义药物的GCNLayer结构
        self.catFeature = OnlyConcatFeature(device)     #!!
        # self.catAdjMatrix = OnlyConcatMatrix(self.cell_adj_matrix,self.drug_adj_matrix,device)     #!!

        # GraphUNet 编码解码结构
        self.graph_unet = GraphUnet(ks, in_dim=n_hid1, out_dim=n_hid2, dim=dim, act=act, drop_p=drop_p)
        #原来的GCN结构
        self.gcnEncoder = GEncoder(self.adj_matrix_hat,self.feature,460)
        # 解码器拆分特征矩阵
        self.decoder = GDecoder(adj_mat.shape[0], adj_mat.shape[1], n_hid1=384, n_hid2=72, alpha=alpha,device=device)
    def forward(self):
        gcnCellFeature = self.GCNCellFeature(self.cell_adj_matrix, self.cellFeature)     #!!
        gcnDrugFeature = self.GCNDrugFeature(self.drug_adj_matrix, self.drugFeature)     # !!原方法，使用子结构指纹计算药物的同构GCN特征
        # gcnDrugFeature = self.GCNDrugFeature(self.drug_adj_matrix, self.drug_graph_feature)  # !!数据层面的消融实验-使用药物分子图代替药物的子结构指纹
        gcnDrugFeature = torch.add(gcnDrugFeature, self.drug_graph_feature)   # 数据层面的消融实验-去掉药物的分子图特征
        # concatFeature = self.catFeature(gcnCellFeature, gcnDrugFeature)     #!!
        # h = self.graph_unet(self.adj_matrix_hat, concatFeature)     #!!

        # h = self.graph_unet(self.adj_matrix_hat, self.feature)   # 使用GraphUNet对异构图进行特征编码
        h = self.graph_unet(self.Unet_adj_matrix, self.feature)   # 使用GraphUNet对异构图进行特征编码，更换了邻接矩阵

        # 使用GCN对异构图进行特征编码
        encode_output = self.gcnEncoder()
        # output = h  # 使用最后一层的输出作为最终输出
        # output = self.decoder(h, self.cellFeature, self.drugFeature)
        # output = self.decoder(h, gcnCellFeature, gcnDrugFeature)
        output = self.decoder(h, encode_output, gcnCellFeature, gcnDrugFeature)  #总的结构

        # output = self.decoder(h, encode_output, gcnDrugFeature)     #消融实验之去掉细胞系的同构GCN分支
        # output = self.decoder(h, encode_output, gcnCellFeature)     #消融实验之去掉药物的同构GCN分支
        # output = self.decoder(h, encode_output)     #消融实验之同时去掉细胞系和药物的同构GCN分支
        # output = self.decoder(encode_output, gcnCellFeature, gcnDrugFeature)    #消融实验去掉GUN分支
        # output = self.decoder(h, gcnCellFeature, gcnDrugFeature)    #消融实验去掉异构GCN分支

        # output = self.decoder(h)
        return output

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
        self.len += 1

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
