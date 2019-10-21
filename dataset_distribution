from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import csv
import os

def default_loader(path):
    return Image.open(path).convert('RGB')

transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    # transforms.RandomCrop(96),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class MyDataset(Dataset):
    def __init__(self, csv_file_path, transform=transforms, loader=default_loader):
        imgs = list()
        csv_file = csv.reader(open(csv_file_path, 'r'))
        for line in csv_file:
            if len(line[1]) <= 2:
                imgs.append(line)
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        # img = self.loader('datasets/Train/'+ filename)
        # if self.transform is not None:
        #     img = self.transform(img)
        label = int(label) - 1

        return filename, label

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    train_data = MyDataset('datasets/Train_label.csv')
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=1,
                          pin_memory=True)

    labels = []
    for i, (image_path, label) in enumerate(train_loader, 0):
        image_path = image_path[0]
        label = label.numpy().tolist()[0]
        # print(image_path, label)
        labels.append(label)

    class_count = {}
    for i in set(labels):
        class_count[i] = labels.count(i)
    res = sorted(class_count.items(), key=lambda class_count: class_count[1], reverse=True)

    _class = []
    _num = []
    for ele in res:
        _class.append(ele[0])
        _num.append(ele[1])

    classes = ["中云-高积云-絮状高积云", "中云-高积云-透光高积云",
               "中云-高积云-荚状高积云", "中云-高积云-积云性高积云",
               "中云-高积云-蔽光高积云", "中云-高积云-堡状高积云",
               "中云-高层云-透光高层云", "中云-高层云-蔽光高层云",
               "高云-卷云-伪卷云", "高云-卷云-密卷云",
               "高云-卷云-毛卷云", "高云-卷云-钩卷云",
               "高云-卷积云-卷积云", "高云-卷层云-匀卷层云",
               "高云-卷层云-毛卷层云", "低云-雨层云-雨层云",
               "低云-雨层云-碎雨云", "低云-积云-碎积云",
               "低云-积云-浓积云", "低云-积云-淡积云",
               "低云-积雨云-鬃积雨云", "低云-积雨云-秃积雨云",
               "低云-层云-碎层云", "低云-层云-层云",
               "低云-层积云-透光层积云", "低云-层积云-荚状层积云",
               "低云-层积云-积云性层积云", "低云-层积云-蔽光层积云",
               "低云-层积云-堡状层积云"]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 6), dpi=80)

    # 再创建一个规格为 1 x 1 的子图
    plt.subplot(1, 1, 1)

    N = 29

    values = _num
    # index = [classes[i] for i in _class]
    index = [i for i in _class]

    plt.axes(aspect=1)
    plt.pie(x=values, labels=index, autopct='%.0f%%')

    # 添加标题
    plt.title('Distributions')

    plt.show()
