#只是提供网络
import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
from dataset import *
from torch import optim
from torch.utils.data import DataLoader
import torch.jit as jit


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        #torch.Size([512, 108])
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)),requires_grad=True)
        self.func = nn.Softmax()

    def forward(self, x, s=64, m=0.5):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / s
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))

        return arcsoftmax


class FaceNet(nn.Module):

    def __init__(self):
        super(FaceNet, self).__init__()
        self.sub_net = nn.Sequential(
            models.mobilenet_v2(),

        )
        # print(models.mobilenet_v2())
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            # nn.PReLU(),
            nn.Linear(1000, 512, bias=False),
        )
        self.arc_softmax = Arcsoftmax(512,108)

    def forward(self, x):
        y = self.sub_net(x)
        feature = self.feature_net(y)

        return feature, self.arc_softmax(feature)

    def encode(self, x):
        return self.feature_net(self.sub_net(x))


def compare(face1, face2):
    print(face1.shape,face2.shape)
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)
    # print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.T)
    # cosb = torch.dot(face1_norm.reshape(-1), face2_norm.reshape(-1))
    return cosa




if __name__ == '__main__':

    # 训练过程
    # net = FaceNet().cuda()
    # device = torch.device('cuda')
    # #loss(input, class) = -input[class]。举个例，三分类任务，
    # # input=[-1.233, 2.657, 0.534]， 真实标签为2（class=2），则loss为-0.534。就是对应类别上的输出，取一个负号！
    # loss_fn = nn.NLLLoss()
    # optimizer = optim.Adam(net.parameters())
    # # dataset = MyDataset(r"data3")
    # dataset = MyDataset(r"D:\程序代码\数据集\明星脸\明星脸part")
    # dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)
    #
    # for epoch in range(1000):
    #     predict = torch.tensor([]).cuda()
    #     label = torch.tensor([]).cuda()
    #     for xs, ys in dataloader:
    #
    #         feature, cls = net(xs.cuda())
    #         loss = loss_fn(torch.log(cls), ys.cuda())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         ys = ys.to(device, dtype=torch.float32)
    #         predict = torch.cat((predict,cls),dim=0)
    #         # print(predict, ys.cuda())
    #         # ys = torch.tensor(ys.cuda(), dtype=torch.float32)
    #         label =  torch.cat((label,ys),dim=0)
    #         # label.extend(ys)
    #     # print(type(predict),type(label))
    #     # exit()
    #     print(torch.argmax(predict, dim=1), label,len(label))
    #     # print(torch.sum(torch.argmax(cls, dim=1)==ys.cuda()))
    #     # print(str(epoch)+"Loss====>"+str(loss.item()))
    #     acc = torch.div(torch.sum(torch.argmax(predict, dim=1)==label).float(),len(label))
    #     print(str(epoch) + "acc====>" + str(acc.item()))
    #     if epoch%10==0:
    #         torch.save(net.state_dict(), "params/1.pt")
    #         print(str(epoch)+"参数保存成功")

    # 使用
    net = FaceNet().cuda()
    net.load_state_dict(torch.load("param/1.pt"))
    net.eval()
    tensor = net.arc_softmax.w
    print(tensor.T.shape)
    exit()
    person1 = tf(Image.open("test_img/pic146.jpg")).cuda()
    person1_feature = net.encode(torch.unsqueeze(person1,0))
    # print(person1_feature.shape)
    # exit()
    # person1_feature = net.encode(person1[None, ...])
    # print(person1.shape)-
    # print(torch.unsqueeze(person1,0).shape)
    # print(person1[None, ...].shape)

    person2 = tf(Image.open("test_img/pic146.jpg")).cuda()
    # person2 = tf(Image.open("test_img/1.bmp")).cuda()
    person2_feature = net.encode(person2[None, ...])

    siam = compare(person1_feature, person2_feature)
    print(max(0,siam.item()))
