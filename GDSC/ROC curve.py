import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd

# 模拟数据：真实标签和各个模型的预测分数
y_true = pd.read_csv('true.csv', header=None).values.flatten()  # 确保是一个一维数组

# 模拟各个模型的预测分数
y_scores_srmf = pd.read_csv('SRMF.csv', header=None).values.flatten()
y_scores_hnmrdp = pd.read_csv('HNMDRP.csv', header=None).values.flatten()
y_scores_deepcdr = pd.read_csv('DeepCDR.csv', header=None).values.flatten()
y_scores_graphcdr = pd.read_csv('GraphCDR.csv', header=None).values.flatten()
y_scores_bandrp = pd.read_csv('BANDRP.csv', header=None).values.flatten()
y_scores_mofgcn = pd.read_csv('MOFGCN.csv', header=None).values.flatten()
y_scores_hhhcdr = pd.read_csv('HHHCDR.csv', header=None).values.flatten()
# 创建图形
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# ROC 曲线
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

# 计算 ROC 曲线和 AUC
fpr_dict['SRMF'], tpr_dict['SRMF'], _ = roc_curve(y_true, y_scores_srmf)
fpr_dict['HNMDRP'], tpr_dict['HNMDRP'], _ = roc_curve(y_true, y_scores_hnmrdp)
fpr_dict['DeepCDR'], tpr_dict['DeepCDR'], _ = roc_curve(y_true, y_scores_deepcdr)
fpr_dict['GraphCDR'], tpr_dict['GraphCDR'], _ = roc_curve(y_true, y_scores_graphcdr)
fpr_dict['BANDRP'], tpr_dict['BANDRP'], _ = roc_curve(y_true, y_scores_bandrp)
fpr_dict['H3CDR'], tpr_dict['H3CDR'], _ = roc_curve(y_true, y_scores_mofgcn)
fpr_dict['H^3CDR'], tpr_dict['H^3CDR'], _ = roc_curve(y_true, y_scores_hhhcdr)

roc_auc_dict['SRMF'] = auc(fpr_dict['SRMF'], tpr_dict['SRMF'])
roc_auc_dict['HNMDRP'] = auc(fpr_dict['HNMDRP'], tpr_dict['HNMDRP'])
roc_auc_dict['DeepCDR'] = auc(fpr_dict['DeepCDR'], tpr_dict['DeepCDR'])
roc_auc_dict['GraphCDR'] = auc(fpr_dict['GraphCDR'], tpr_dict['GraphCDR'])
roc_auc_dict['BANDRP'] = auc(fpr_dict['BANDRP'], tpr_dict['BANDRP'])
roc_auc_dict['H3CDR'] = auc(fpr_dict['H3CDR'], tpr_dict['H3CDR'])
roc_auc_dict['H^3CDR'] = auc(fpr_dict['H^3CDR'], tpr_dict['H^3CDR'])

# 绘制 ROC 曲线
for model in fpr_dict:
    # 使用 LaTeX 格式化 "H^3CDR" 为带上标的格式
    if model == 'H^3CDR':
        axs[0].plot(fpr_dict[model], tpr_dict[model], label=r'$\text{H}^3\text{CDR}$')  # 上标格式
    else:
        axs[0].plot(fpr_dict[model], tpr_dict[model], label=model)

axs[0].plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机分类器的线
axs[0].set_title('ROC Curve', fontsize=18)
axs[0].set_xlabel('False Positive Rate', fontsize=18)
axs[0].set_ylabel('True Positive Rate', fontsize=18)
axs[0].legend(loc='lower right' , fontsize=14)
axs[0].tick_params(axis='y',  labelsize=16)# 增大纵坐标的数字大小
axs[0].tick_params(axis='x', labelsize=16)  # 增大横坐标的数字大小
# PR 曲线
precision_dict = {}
recall_dict = {}
average_precision_dict = {}

# 计算 PR 曲线和 AUC (平均精度)
precision_dict['SRMF'], recall_dict['SRMF'], _ = precision_recall_curve(y_true, y_scores_srmf)
precision_dict['HNMDRP'], recall_dict['HNMDRP'], _ = precision_recall_curve(y_true, y_scores_hnmrdp)
precision_dict['DeepCDR'], recall_dict['DeepCDR'], _ = precision_recall_curve(y_true, y_scores_deepcdr)
precision_dict['GraphCDR'], recall_dict['GraphCDR'], _ = precision_recall_curve(y_true, y_scores_graphcdr)
precision_dict['BANDRP'], recall_dict['BANDRP'], _ = precision_recall_curve(y_true, y_scores_bandrp)
precision_dict['H3CDR'], recall_dict['H3CDR'], _ = precision_recall_curve(y_true, y_scores_mofgcn)
precision_dict['H^3CDR'], recall_dict['H^3CDR'], _ = precision_recall_curve(y_true, y_scores_hhhcdr)

average_precision_dict['SRMF'] = average_precision_score(y_true, y_scores_srmf)
average_precision_dict['HNMDRP'] = average_precision_score(y_true, y_scores_hnmrdp)
average_precision_dict['DeepCDR'] = average_precision_score(y_true, y_scores_deepcdr)
average_precision_dict['GraphCDR'] = average_precision_score(y_true, y_scores_graphcdr)
average_precision_dict['BANDRP'] = average_precision_score(y_true, y_scores_bandrp)
average_precision_dict['H3CDR'] = average_precision_score(y_true, y_scores_mofgcn)
average_precision_dict['H^3CDR'] = average_precision_score(y_true, y_scores_hhhcdr)

# 绘制 PR 曲线
for model in precision_dict:
    if model == 'H^3CDR':
        axs[1].plot(recall_dict[model], precision_dict[model], label=r'$\text{H}^3\text{CDR}$')
    else:
        axs[1].plot(recall_dict[model], precision_dict[model], label=model)


axs[1].set_title('Precision-Recall Curve', fontsize=18)
axs[1].set_xlabel('Recall', fontsize=18)
axs[1].set_ylabel('Precision', fontsize=18)
axs[1].legend(loc='lower left', fontsize=14)
axs[1].tick_params(axis='y',  labelsize=16)# 增大纵坐标的数字大小
axs[1].tick_params(axis='x',  labelsize=16)  # 增大横坐标的数字大小
# 调整布局并显示
plt.tight_layout()
plt.show()
