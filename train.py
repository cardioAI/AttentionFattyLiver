import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from my_dataset import MyDataSet
from model import ImpedanceNet
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['time', 'step', 'running_loss', 'val_accurate'])
df.to_csv('./train_acc.csv', index=False)


def get_list(impedance_url):
    impedance_path = []
    impedance_class = []
    for class_name in os.listdir(impedance_url):
        for file_npy in os.listdir(impedance_url + '/' + class_name):
            impedance_path.append(impedance_url + '/' + class_name + '/' + file_npy)
            impedance_class.append(class_name)
    return impedance_path, impedance_class


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    data_root = os.path.abspath(os.path.join(os.getcwd(), 'fatty_liver_dataset'))

    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)
    train_impedance_path, train_impedance_class = get_list(data_root + '/train')
    train_dataset = MyDataSet(train_impedance_path, train_impedance_class)
    train_num = len(train_dataset)
    # print(train_num)
    train_loader = torch.utils.data.DataLoader(train_dataset)

    val_impedance_path, val_impedance_class = get_list(data_root + '/train')
    val_dataset = MyDataSet(val_impedance_path, val_impedance_class)
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset)

    # print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = ImpedanceNet()
    # print(net)
    net = net.float()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 500
    save_path = './ImpedanceNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    # print(train_steps)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            impedances, labels = data
            optimizer.zero_grad()
            impedances = impedances.float()
            outputs = net(impedances.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_impedances, val_labels = val_data
                val_impedances = val_impedances.float()
                outputs = net(val_impedances.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        time = "%s" % datetime.now()
        list = [time, epoch, running_loss, val_accurate]
        data = pd.DataFrame([list])
        data.to_csv('./train_acc.csv', mode='a', header=False, index=False)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Finished Training')

    data = pd.read_csv('./train_acc.csv')
    data_loss = data[['running_loss']]  # class 'pandas.core.frame.DataFrame'
    data_acc = data[['val_accurate']]
    x = np.arange(0, epochs, 1)
    y1 = np.array(data_loss)  # 将DataFrame类型转化为numpy数组
    y2 = np.array(data_acc)
    # 绘图
    plt.plot(x, y1, label="loss")
    plt.plot(x, y2, label="accuracy")
    plt.title("loss & accuracy")
    plt.xlabel('epoch')
    plt.ylabel('probability')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
