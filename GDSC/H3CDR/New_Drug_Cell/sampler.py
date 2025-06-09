import torch
import numpy as np
import pandas as pd

class Sampler(object):
    def __init__(self, original_adj_mat, null_mask, target_dim, target_index):
        super(Sampler, self).__init__()
        self.adj_mat = original_adj_mat
        self.null_mask = null_mask
        self.dim = target_dim
        self.target_index = target_index
        self.train_data, self.test_data = self.sample_train_test_data()
        self.train_mask, self.test_mask = self.sample_train_test_mask()


    def sample_target_test_index(self):
        if self.dim:
            # target_pos_index = self.adj_mat[:,self.target_index].astype(np.bool_)
            target_pos_index = np.where(self.adj_mat[:, self.target_index] == 1)[0]
        else:
            target_pos_index = np.where(self.adj_mat[self.target_index, :] == 1)[0]          #返回了一个数组，记录着邻接矩阵中第target_index行里元素为1的下标
        return target_pos_index

    def sample_train_test_data(self):
        test_data = np.zeros(self.adj_mat.shape, dtype=np.float32)
        test_index = self.sample_target_test_index()
        if self.dim:
            test_data[test_index, self.target_index] = 1
        else:
            test_data[self.target_index, test_index] = 1                      #test_data里边是第target_index行的反应情况
        train_data = self.adj_mat - test_data                           #train_data中是把adj里边有关第target_index行的反应情况给去掉了
        train_data = torch.from_numpy(train_data)
        test_data = torch.from_numpy(test_data)
        return train_data, test_data

    def sample_train_test_mask(self):
        test_index = self.sample_target_test_index()
        neg_value = np.ones(self.adj_mat.shape, dtype=np.float32)
        neg_value = neg_value - self.adj_mat - self.null_mask                  #neg_value中 0表示敏感/尚未实验；1表示耐药。
        neg_test_mask = np.zeros(self.adj_mat.shape, dtype=np.float32)
        if self.dim:
            target_neg_index = np.where(neg_value[:, self.target_index] == 1)[0]
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(target_neg_index, test_index.shape[0], replace=False)
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[target_neg_test_index, self.target_index] = 1
            neg_value[:, self.target_index] = 0
        else:
            target_neg_index = np.where(neg_value[self.target_index, :] == 1)[0]         #target_neg_index是把目标行中耐药的下标给记录下来
            if test_index.shape[0] < target_neg_index.shape[0]:
                target_neg_test_index = np.random.choice(target_neg_index, test_index.shape[0], replace=False)
            else:
                target_neg_test_index = target_neg_index
            neg_test_mask[self.target_index, target_neg_test_index] = 1        #neg_test_mask记录着 目标行中与敏感样本数量相同的耐药样本的下标
            neg_value[self.target_index, :] = 0
        train_mask = (self.train_data.numpy() + neg_value).astype(np.bool_)     #train_mask中1表示敏感或耐药；0表示尚未实验
        test_mask = (self.test_data.numpy() + neg_test_mask).astype(np.bool_)        #test_mask中标记测试样本，其中正样本与负样本数量相同


        # indices = np.where(test_mask[:, 57] == 1)[0]  # 获取第58列为1的所有行下标
        indices = np.where(test_mask[:, 50] == 1)[0]  # 获取第58列为1的所有行下标
        #记录测试集的下标
        # 将 indices 作为一行添加到 DataFrame 中
        indices_df = pd.DataFrame([indices])
        file_path = './result_data/drug_50_true_data_index.csv'
        # 如果文件已经存在，则追加数据；否则创建新文件
        try:
            # 尝试读取现有的 CSV 文件
            existing_df = pd.read_csv(file_path, header=None)
            # 将新的一行下标添加到现有的 DataFrame
            all_indices_df = pd.concat([existing_df, indices_df], ignore_index=True)
        except FileNotFoundError:
            # 如果文件不存在，直接创建一个新的 DataFrame
            all_indices_df = indices_df
        # 最后将合并后的 DataFrame 保存到 CSV 文件
        all_indices_df.to_csv(file_path, index=False, header=False)


        train_mask = torch.from_numpy(train_mask)
        test_mask = torch.from_numpy(test_mask)
        return train_mask, test_mask
