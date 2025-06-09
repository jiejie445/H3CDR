from GDSC.H3CDR.model import GModel
from GDSC.H3CDR.optimizer import Optimizer
from GDSC.H3CDR.New_Drug_Cell.sampler import Sampler
from GDSC.H3CDR.myutils import (roc_auc, evaluate_metrics, translate_result)

def mofgcn_new_target(gene, cna, mutation, drug_feature, drug_graph_feature, response_mat, null_mask, target_dim, target_index,
                      evaluate_fun, sigma=2, knn=11, iterates=3, n_hid1=192, n_hid2=36, alpha=5.74, lr=5e-4,
                      epochs=1000, device="cuda:0"):
    """
    :param sigma: an scale parameter, int or float el.
    :param knn: KNN parameter, int
    :param iterates: iterate parameter, int
    :param n_hid1: the frist hiden layer, int
    :param n_hid2: the second hiden layer, int
    :param alpha: a scale parameter
    :param lr: learning rate, float
    :param epochs: apochs, int
    :param gene: cell gene feature, narray
    :param cna: cell cna feature, narray
    :param mutation:cell mutation feature, narray
    :param drug_feature: drug fingerprint feature, narray
    :param response_mat: response matrix, narray
    :param null_mask: null mask of response_mat, narray
    :param target_dim: drug-1 or cell-0, int
    :param target_index: target index in response matrix, int scale
    :param evaluate_fun: evaluate function
    :param device: run device, cpu or cuda:0
    :return: AUC, ACC, F1-score and so on, an scalar, score
    """
    sampler = Sampler(response_mat, null_mask, target_dim, target_index)
    model = GModel(sampler.train_data, gene=gene, cna=cna, mutation=mutation, null_mask=null_mask, sigma=sigma, k=knn, iterates=iterates,
                   feature_drug=drug_feature, graph_feature_drug=drug_graph_feature, n_hid1=1190, n_hid2=384,dim=1190, alpha=alpha, device=device)
    opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask, evaluate_metrics,
                    lr=lr, epochs=epochs, device=device)
    epoch, true_data, predict_data = opt()
    return epoch, true_data, predict_data
