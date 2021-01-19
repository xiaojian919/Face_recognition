# import torch
#
# a = torch.randn([1,512])
# b = torch.matmul(torch.nn.functional.normalize(a),torch.nn.functional.normalize(a).t())
# c = torch.dot(torch.nn.functional.normalize(a).reshape(-1),torch.nn.functional.normalize(a).reshape(-1))
# print(b)
# print(c)


# import torch
# import numpy as np
#
# a=np.array([[2],[3],[4],[6]])
# a=torch.from_numpy(a)  ####将numpy 转化为tensor
# b=np.array([[2],[3],[4],[3]])
# b=torch.from_numpy(b)
# c = a == b
# d = torch.sum(c)
# acc = torch.div(torch.sum(a==b).float(),len(b))
# print(acc)
# print(len(c))
# print(torch.equal(a,b))
# for i in range(0,a.shape[0]):
#     print(a[i],b[i])
#     if (torch.equal(a[i],b[i])):
#         print("66666")



# 使用

import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
from dataset import *
from face import FaceNet


def Arcsoftmax(x,y, s=64, m=0.5):
    x_norm = F.normalize(x, dim=1)
    y_norm = F.normalize(y, dim=0)

    cosa = torch.matmul(x_norm, y_norm) / s
    a = torch.acos(cosa)

    arcsoftmax = torch.exp(
        s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
        s * cosa) + torch.exp(s * torch.cos(a + m)))

    return arcsoftmax,1

if __name__ == '__main__':

    net = FaceNet().cuda()
    net.load_state_dict(torch.load("param/1.pt"))
    net.eval()
    tensor = net.arc_softmax.w
    print(tensor.T.shape)

    person1 = tf(Image.open("test_img/1.jpg").convert('RGB')).cuda()
    person1_feature = net.encode(torch.unsqueeze(person1, 0))
    print(person1_feature.shape)
    add = torch.cat((tensor.T, person1_feature), dim=0)
    print(add.shape)
    a,b = Arcsoftmax(person1_feature,add.T)
    # print(a)
    # x_norm = F.normalize(person1_feature, dim=1)
    # #对单个特征向量标准化
    # y_norm = F.normalize(add, dim=1)
    # torch.save(y_norm,"test")
    # a = torch.matmul(x_norm, y_norm.T)
    #
    # print(a.shape)
    # cls = torch.argmax(a, dim=1)
    # print(cls.item())
    # print(a)

    # torch.save(Arcsoftmax(person1_feature,add.T),"1.pth")
    # a,b = torch.load("1.pth")
    # print("a={}".format(a))
    # print("b={}".format(b))



