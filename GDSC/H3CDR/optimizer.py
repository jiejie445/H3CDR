import torch
from abc import ABC
import torch.nn as nn
import torch.optim as optim
from model import EarlyStop
from myutils import cross_entropy_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Optimizer(nn.Module, ABC):
    def __init__(self, model, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.01, epochs=200, test_freq=20, device="gpu"):
        super(Optimizer, self).__init__()
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun        # 此处传入的 evaluate_fun 应为多个指标的集合
        self.lr = lr
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        early_stop = EarlyStop(tolerance=8, data_len=true_data.size()[0])
        # print("true_data的形状：",true_data.shape)
        for epoch in torch.arange(self.epochs):
            predict_data = self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            self.optimizer.step()

            if epoch % self.test_freq == 0:
                predict_data_masked = torch.masked_select(predict_data, self.test_mask)
                # print("predict_data_masked的形状：", predict_data_masked.shape)
                # 使用传入的 evaluate_fun 计算多个评价指标
                results = self.evaluate_fun(true_data, predict_data_masked)
                # 如果 evaluate_fun 返回多个结果，输出它们
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), end=" ")
                for metric_name, metric_value in results.items():
                    print(f"{metric_name}: {metric_value:.4f}", end=" ")
                print("")
                # 早停判断
                flag = early_stop.stop(auc=results['AUC'], epoch=epoch.item(), predict_data=predict_data_masked)
                if flag:
                    break

        print("Fit finished.")
        max_index = early_stop.get_best_index()
        best_epoch = early_stop.epoch_pre[max_index]
        best_predict = early_stop.predict_data_pre[max_index, :]

        # 文件路径
        true_data_file = 'true_data.csv'
        best_predict_file = 'best_predict.csv'
        true_data_example = true_data
        best_predict_example = best_predict
        # 将 true_data 和 best_predict 转换为 NumPy 数组
        true_data_numpy = true_data_example.cpu().numpy()  # 使用 .cpu() 将张量从 CUDA 设备转移到 CPU 上
        best_predict_numpy = best_predict_example.detach().cpu().numpy()
        # 将数据追加到文件中
        def save_to_csv(data, file_path):
            # 将一维数据转化为二维数据（每个元素为一行）
            data = np.expand_dims(data, axis=0)  # 这样数据的形状将变成 (1, n)

            # 如果文件不存在，创建文件并写入数据；如果文件已存在，则追加数据
            try:
                df = pd.DataFrame(data)
                df.to_csv(file_path, mode='a', header=False, index=False)  # 不保存行索引，不保存列头
            except FileNotFoundError:
                # 如果文件不存在，则先创建一个文件并写入列头
                df = pd.DataFrame(data)
                df.to_csv(file_path, mode='w', header=True, index=False)  # 保存列头

        print("true_data_numpy：",true_data_numpy.shape)
        print("best_predict_numpy：",best_predict_numpy.shape)
        # 保存 true_data 和 best_predict 到 CSV 文件
        save_to_csv(true_data_numpy, true_data_file)
        save_to_csv(best_predict_numpy, best_predict_file)


        return best_epoch, true_data, best_predict
#  1. 扁平化矩阵B并获取最大值的20个下标
#             flat_B = predict_data.flatten()  # 将B矩阵展平为一维数组
#             top_20_indices = np.argsort(flat_B)[-20:]  # 获取最大20个概率值的索引