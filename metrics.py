from torchmetrics import ROC, Accuracy, ConfusionMatrix
import torch
import numpy as np
from model import ImpedanceNet
import os
os.environ["KMP_DPULICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')


class ROC_ConfusionMatrix():
    model = ImpedanceNet()
    state = torch.load(os.path.join(os.getcwd(), 'ImpedanceNet.pth'))
    model.load_state_dict(state)
    url = os.path.join(os.getcwd(), "fatty_liver_dataset/test")
    num_classes = 2
    labels = [0, 1]
    l = 6
    l1 = [0]*l
    l2 = [1]*l
    target = l1 + l2
    y_true = np.array(target)
    # print(y_true)
    y_predit = []
    scores = []

    def compute_accuracy(y, y_pred):

        num = y.shape[0]
        num_correct = np.sum(y_pred == y)
        acc = float(num_correct) / num
        return acc

    url = os.path.join(os.getcwd(), 'fatty_liver_dataset/test')

    for i in os.listdir(url):
        impedance = np.load(os.path.join(url, i))
        impedance = torch.from_numpy(impedance).unsqueeze(0)
        impedance = impedance.unsqueeze(0)

        model.eval()
        out = model(impedance.float())
        _, pred = torch.max(out, 1)
        # print(pred)
        score = torch.softmax(out, dim=1)  # out = model(data)
        score = score.detach().numpy().flatten()
        scores.append(score[1])
        # print(scores)

    # print(scores)

    target = torch.tensor(y_true)
    pred = torch.tensor(scores)
    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(pred, target)

    # best threshold
    thresh = thresholds[torch.argmax(tpr - fpr)]

    # accuracy under best threshold
    y_pred = (pred > thresh)
    accuracy = Accuracy(task="binary")
    acc = accuracy(pred, target)

    confmat = ConfusionMatrix(task="binary", num_classes=2)

    conf_max = confmat(pred, target)
    print(fpr)
    print(tpr)
    print(thresholds)
    print(thresh)
    print(acc)
    print(conf_max)



if __name__ == '__main__':
    roc_auc = ROC_ConfusionMatrix()

