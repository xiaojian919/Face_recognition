from face import FaceNet
import torch
from torchvision import transforms
from PIL import Image,ImageFont
import os







class recognition:
    def __init__(self,net,param):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net = net.to(self.device)
        self.net.load_state_dict(torch.load(param))
        self.w = self.net.arc_softmax.w
        self.wzeros = torch.zeros([512, 0]).cuda()
        self.net.eval()
        self.tf = transforms.Compose([
            transforms.Resize([112,112]),
            transforms.ToTensor(),
            ])
    def addfeature(self,imgs):
        names = []
        for personname in os.listdir(imgs):
                for img in os.listdir(os.path.join(imgs, personname)):
                    person = self.tf(Image.open(os.path.join(imgs,personname,img)).convert('RGB')).cuda()
                    person_feature = self.net.encode(torch.unsqueeze(person, 0))
        #     self.w = torch.cat((self.w.T, person_feature), dim=0)
        #     self.w = self.w.T
        # return self.w
                    self.wzeros = torch.cat((self.wzeros.T, person_feature), dim=0)
                    self.wzeros = self.wzeros.T
                    names.append(personname)
        return self.wzeros,names

if __name__ == '__main__':
    # 网络初始化
    net = FaceNet()
    param = "param/1.pt"
    rec = recognition(net, param)
    # 扩充特征库
    addimgs = r"./addface"
    torch.save(rec.addfeature(addimgs),"w.pth")
    w, names = torch.load("w.pth")
    print(w.shape)
    print(names)

