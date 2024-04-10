import torch as t
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class ACC(object):
    def __init__(self):
        super(ACC, self).__init__()
        self.name = 'ACC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        total = len(answer)
        answer = t.Tensor(answer)
        label = t.Tensor(label)
        pred = t.argmax(answer, dim=1)
        correct = sum(pred == label).float()
        acc = correct / total
        return acc.item()


class AUC(object):
    def __init__(self):
        super(AUC, self).__init__()
        self.name = 'AUC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer)
        answer = answer[:,1]
        answer = answer.tolist()
        result = roc_auc_score(y_true = label, y_score= answer)
        return result


class MAE(object):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer).squeeze(-1)
        label = t.Tensor(label)
        MAE = F.l1_loss(answer, label, reduction = 'mean')
        return MAE.item()

class RMSE(object):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = 'RMSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer).squeeze(-1)
        label = t.Tensor(label)
        RMSE = F.mse_loss(answer, label, reduction = 'mean').sqrt()
        return RMSE.item()
