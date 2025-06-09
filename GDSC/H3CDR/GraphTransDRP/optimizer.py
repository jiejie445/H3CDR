import torch
from abc import ABC
import torch.nn as nn
import torch.optim as optim
from model import EarlyStop
from myutils import cross_entropy_loss
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

        for epoch in torch.arange(self.epochs):
            # 启用异常检测
            torch.autograd.set_detect_anomaly(True)

            predict_data = self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            # loss.backward()
            print("Before backward, is the computation graph retained?", torch.is_grad_enabled())
            loss.backward(retain_graph=True)
            print("After backward, is the computation graph retained?", torch.is_grad_enabled())
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            self.optimizer.step()

            if epoch % self.test_freq == 0:
                predict_data_masked = torch.masked_select(predict_data, self.test_mask)
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
        return best_epoch, true_data, best_predict