# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
eps=np.finfo(float).eps



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores, cls_iu, m_1 = cm2score(self.sum)
        scores.update(cls_iu)
        scores.update(m_1)
        return scores


def cm2score(confusion_matrix):
    # ---------------------------------------------------------------------- #
    # 0. hist
    # [tn, fp      --->sum_a0=[tn+fn, fp+tp]
    #  fn, tp]     --->sum_a1=[tn+fp, fn+tp]
    # ---------------------------------------------------------------------- #
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist) #[tn, tp]
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    # overall acc
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    # recall
    acc_cls_ = tp / (sum_a1 + np.finfo(np.float32).eps) # [tn/(tn+fp), tp/(fn+tp)]

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps) # [tn/(tn+fn), tp/(fp+tp)]

    # F1 score
    F1 = 2*acc_cls_ * precision / (acc_cls_ + precision + np.finfo(np.float32).eps)
    # ---------------------------------------------------------------------- #
    # 2. Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    pc2 = sum_a0 / (hist.sum() + np.finfo(np.float32).eps)# [(tn+fn)/num, (fp+tp)/num] ->[pcn2,pcp2]
    pc1 = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)# [(tn+fp)/num, (fn+tp)/num] ->[pcn1,pcp1]
    pc = sum(pc2*pc1)
    kappa = (acc - pc)/(1-pc)

    cls_iu = dict(zip(range(n_class), iu))



    return {'Overall_Acc': acc,
            'Mean_IoU': mean_iu}, cls_iu, \
           {
        'precision_1': precision[1],
        'recall_1': acc_cls_[1],
        'F1_1': F1[1],
        'Kappa':kappa,}


class RunningMetrics(object):
    def __init__(self, num_classes):
        """
        Computes and stores the Metric values from Confusion Matrix
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param num_classes: <int> number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def __fast_hist(self, label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < self.num_classes)
        hist = np.bincount(self.num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_gts, label_preds):
        """
        Compute Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gts: <np.ndarray> ground-truths
        :param label_preds: <np.ndarray> predictions
        :return:
        """
        for lt, lp in zip(label_gts, label_preds):
            self.confusion_matrix += self.__fast_hist(lt.flatten(), lp.flatten())

    def reset(self):
        """
        Reset Confusion Matrix
        :return:
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_cm(self):
        return self.confusion_matrix

    def get_scores(self):
        """
        Returns score about:
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :return:
        """
        hist = self.confusion_matrix
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)

        # ---------------------------------------------------------------------- #
        # 1. Accuracy & Class Accuracy
        # ---------------------------------------------------------------------- #
        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        # recall
        acc_cls_ = tp / (sum_a1 + np.finfo(np.float32).eps)

        # precision
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)
        # ---------------------------------------------------------------------- #
        # 2. Mean IoU
        # ---------------------------------------------------------------------- #
        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        cls_iu = dict(zip(range(self.num_classes), iu))

        # F1 score
        F1 = 2 * acc_cls_ * precision / (acc_cls_ + precision + np.finfo(np.float32).eps)

        # kappa
        kappa = cal_kappa(hist)

        scores = {'Overall_Acc': acc,
                'Mean_IoU': mean_iu}
        scores.update(cls_iu)
        scores.update({'precision_1': precision[1],
                       'recall_1': acc_cls_[1],
                       'F1_1': F1[1],
                       'Kappa': kappa},
                       )
        return scores

def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa