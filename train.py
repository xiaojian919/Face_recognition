from dataset import MyDataset
from torch import nn
import torch
from dataset import *
from face import FaceNet
import torch.optim as optim

class Trainer:
    def __init__(self,net,dataset_path,valdataset_path,save_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        #形参变实参
        self.net = net.to(self.device)
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.valdataset_path = valdataset_path
        # loss(input, class) = -input[class]。举个例，三分类任务，
        # input=[-1.233, 2.657, 0.534]， 真实标签为2（class=2），则loss为-0.534。就是对应类别上的输出，取一个负号！
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(net.parameters())

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
        else:
            print("NO Param")



    # dataset = MyDataset(r"data3")
    def train(self,round_limit = 10):
        faceDataset = MyDataset(self.dataset_path)
        valDataset = MyDataset(self.valdataset_path)
        # batch_size一般不要超过百分之一 经验值
        dataloader = DataLoader(faceDataset, batch_size=128, shuffle=True, num_workers=4)
        valdataloader = DataLoader(valDataset, batch_size=32, shuffle=True, num_workers=4)
        acc_end = 0.7457
        epoch = 0
        round = 0
        while True:
            predict = torch.tensor([]).cuda()
            label = torch.tensor([]).cuda()
            for xs, ys in dataloader:
                feature, cls = self.net(xs.cuda())
                loss = self.loss_fn(torch.log(cls), ys.cuda())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("epoch = {}，loss= {}".format(epoch,loss))
            for xs2, ys2 in valdataloader:
                feature2, cls2 = self.net(xs2.cuda())
                loss = self.loss_fn(torch.log(cls2), ys2.cuda())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ys2 = ys2.to(self.device, dtype=torch.float32)
                predict = torch.cat((predict,cls2),dim=0)
                label =  torch.cat((label,ys2),dim=0)
            epoch += 1
            print(torch.argmax(predict, dim=1), label,len(label))
            acc = torch.div(torch.sum(torch.argmax(predict, dim=1)==label).float(),len(label))
            if acc > acc_end:
                torch.save(self.net.state_dict(), self.save_path)
                acc_end = acc
                print("save success，acc更新为{}".format(acc))
                round = 0
            else:
                round += 1
                print("精确度为{},没有提升，参数未更新,acc仍为{},第{}次未更新".format(acc,acc_end,round))
                if round >= round_limit:
                    print("最终acc为{}".format(acc_end))
                    break
            # print(str(epoch) + "acc====>" + str(acc.item()))
            # if epoch%10==0:
            #     torch.save(net.state_dict(), "params/1.pt")
            #     print(str(epoch)+"参数保存成功")

if __name__ == '__main__':
    net = FaceNet()
    dataset_path = r"D:\程序代码\数据集\明星脸\民星脸split\train"
    valdataset_path = r"D:\程序代码\数据集\明星脸\民星脸split\val"
    save_path =  r'.\param\2.pt'
    trainer = Trainer(net, dataset_path, valdataset_path, save_path)
    trainer.train(10)
