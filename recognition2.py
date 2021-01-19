#检测用
from face import FaceNet
import torch
from torchvision import transforms
from torch.nn import functional as F
from MTCNN.detect01 import Detector
from PIL import Image,ImageFont
from PIL import ImageDraw
import numpy as np
import cv2
import os
import time






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
        for img in os.listdir(imgs):
            person = self.tf(Image.open(os.path.join(imgs, img)).convert('RGB')).cuda()
            person_feature = self.net.encode(torch.unsqueeze(person, 0))
        #     self.w = torch.cat((self.w.T, person_feature), dim=0)
        #     self.w = self.w.T
        # return self.w
            self.wzeros = torch.cat((self.wzeros.T, person_feature), dim=0)
            self.wzeros = self.wzeros.T
        return self.wzeros


    def compare(self,img,w):
        person = self.tf(img).cuda()
        person_feature = self.net.encode(torch.unsqueeze(person, 0))
        person_norm = F.normalize(person_feature, dim=1)
        # print(person_norm)
        w_norm = F.normalize(w, dim=0)
        # print(wzeros_norm)
        cls = torch.matmul(person_norm, w_norm)
        # w_norm = F.normalize(self.w, dim=1)
        # cls = torch.matmul(person_norm, w_norm)
        return cls

    def comparetwoface(self,face1, face2):
        person1 = self.tf(Image.open(img).convert('RGB')).cuda()
        person_feature1 = net.encode(torch.unsqueeze(person1, 0))
        person2 = self.tf(Image.open(img).convert('RGB')).cuda()
        person_feature2 = net.encode(torch.unsqueeze(person2, 0))
        face1_norm = F.normalize(person_feature1)
        face2_norm = F.normalize(person_feature2)
        cosa = torch.matmul(face1_norm, face2_norm.T)
        # cosb = torch.dot(face1_norm.reshape(-1), face2_norm.reshape(-1))
        return cosa

if __name__ == '__main__':

    #网络初始化
    net = FaceNet()
    param = "param/1.pt"
    rec = recognition(net,param)
    w,names = torch.load("w.pth")

    #摄像头版本
    # 检测视频
    font_path = "msyh.ttc"
    font = ImageFont.truetype(font_path, size=20)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture('1.mp4')
    # w = int(cap.get(3))
    # h = int(cap.get(4))
    # fps = cap.get(5)
    detector = Detector()

    while True:
        start = time.time()
        ret, frame = cap.read()
        # print(frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif ret == False:
            break
        image = frame[:, :, ::-1]
        image = Image.fromarray(image)
        with torch.no_grad():
            # w, h = image.size
            boxes = detector.detect(image)
            for box in boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if (x2 - x1) > 100 and (y2 - y1) > 100:
                    img = image.crop((x1, y1,x2 ,int(y1+(y2-y1)*0.9)))
                    # img = Image.open(img).convert('RGB')
                    cls = rec.compare(img,w)
                    print(cls)
                    key = torch.argmax(cls, dim=1).item()
                    print(key)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((x1, y1,x2 ,int(y1+(y2-y1)*0.9)), outline='red', width=4)
                    if cls[0][key].item() > 0.95:
                        draw.text((x1 + 10, y1 + 10), text=f'{names[key]}', fill='red',font=font)
                    else:
                        draw.text((x1 + 10, y1 + 10), text=f'{"其他人"}', fill='red', font=font)
            frame = np.array(image)
            end = time.time()
        seconds = end - start
        fps = 1.0 / seconds
        # print(f'FPS:{fps:.2f}')
        cv2.putText(frame, f'FPS:{fps:.2f}', (20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255,0, 0))

        cv2.imshow('x', frame[:, :, ::-1])
    cap.release()
    cv2.destroyAllWindows()


