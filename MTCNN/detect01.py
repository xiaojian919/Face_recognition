import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from MTCNN.tool import utils
from MTCNN import nets_2
from torchvision import transforms
import time
import cv2
import os


class Detector:

    def __init__(self, pnet_param=r"D:\lieweicodetest\MTCNN\param\PRelu\p_net.pth", rnet_param=r"D:\lieweicodetest\MTCNN\param\PRelu\r_net.pth", onet_param=r"D:\lieweicodetest\MTCNN\param\PRelu\o_net.pth",
                 isCuda=True):
    # def __init__(self, pnet_param=r".\temp\p_net.pth", rnet_param=r".\temp\r_net.pth",
    #              onet_param=r".\temp\o_net.pth",isCuda=False):

        self.isCuda = isCuda

        self.pnet = nets_2.Pnet()
        self.rnet = nets_2.Rnet()
        self.onet = nets_2.Onet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param,map_location='cpu'))
        self.rnet.load_state_dict(torch.load(rnet_param,map_location='cpu'))
        self.onet.load_state_dict(torch.load(onet_param,map_location='cpu'))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,],std=[0.5,])
        ])

    def detect(self, image):

        start_time = time.time()

        pnet_boxes = self.__pnet_detect(image)
        # print(pnet_boxes.shape)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # print( rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __pnet_detect(self, img):

        boxes = []
        w, h = img.size
        # print(w,h)
        min_side_len = min(w, h)

        scale = 1

        while min_side_len >= 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            # img_data = torch.unsqueeze(img_data, dim=0)  # 扩维度将[C,H,W]转为[N,C,H,W]
            img_data.unsqueeze_(0)

            _cls, _offest = self.pnet(img_data)
            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            # print(cls.shape) # w h
            # print(offest.shape) #4 w h

            idxs = torch.nonzero(torch.gt(cls, 0.5))
            # print(idxs.shape)
            boxes.extend(self.__box(idxs, offest, cls[idxs[:,0], idxs[:,1]], scale))

            scale *= 0.707
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)
        return utils.nms(np.array(boxes), 0.3)


    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = np.round((start_index[:,1] * stride) / scale)#宽，W，x
        _y1 = np.round((start_index[:,0] * stride) / scale)#高，H,y
        _x2 = np.round((start_index[:,1] * stride + side_len) / scale)
        _y2 = np.round((start_index[:,0] * stride + side_len) / scale)

        ow = _x2 - _x1
        oh = _y2 - _y1


        _offset = offset[:,start_index[:,0], start_index[:,1]]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        box = [x1.numpy(), y1.numpy(), x2.numpy(), y2.numpy(), cls.numpy()]
        box = np.array(box).T


        return box

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset =torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        boxes = []

        idxs, _ = np.where(_cls > 0.8)
        idxs = np.array(idxs)


        _box = _pnet_boxes[idxs]
        # print(_box[0]) #[1291.752    261.65625 1864.8544   834.75867   21.14312]
        _x1 = np.round(_box[:,0])
        _y1 = np.round(_box[:,1])
        _x2 = np.round(_box[:,2])
        _y2 = np.round(_box[:,3])

        ow = _x2 - _x1
        oh = _y2 - _y1
        # print(_x1.shape)
        # print(ow.shape)
        # print(offset[idxs].shape)

        x1 = _x1 + ow * offset[idxs][:,0]
        y1 = _y1 + oh * offset[idxs][:,1]
        x2 = _x2 + ow * offset[idxs][:,2]
        y2 = _y2 + oh * offset[idxs][:,3]
        # _cls =  np.squeeze(_cls)
        cls = _cls[idxs][:, 0]

        boxes.extend((x1, y1, x2, y2, cls))
        boxes = np.array(boxes).T
        # print(boxes.shape)
        # boxes = np.(boxes).T
        # print(boxes.shape)


        return utils.nms(boxes, 0.3)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []

        idxs, _ = np.where(_cls > 0.9999)
        idxs = np.array(idxs)

        _box = _rnet_boxes[idxs]
        _x1 = np.round(_box[:, 0])
        _y1 = np.round(_box[:, 1])
        _x2 = np.round(_box[:, 2])
        _y2 = np.round(_box[:, 3])

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]
        # _cls = np.squeeze(_cls)
        cls = _cls[idxs][:, 0]

        boxes.extend((x1, y1, x2, y2, cls))
        boxes = np.array(boxes).T
        # boxes = boxes.squeeze(axis=0).swapaxes(1,0)#transpose((1, 0))

        return utils.nms(boxes, 0.3, isMin=True)


if __name__ == '__main__':
    # 检测单张
    # x = time.time()
    # with torch.no_grad() as grad:
    #     image_file = r"D:\lieweicodetest\Face_recognition\1.jpg"
    #     detector = Detector()
    #
    #     with Image.open(image_file) as im:
    #         w,h = im.size
    #         im.resize((int(w*0.5),int(h*0.5)))
    #
    #         boxes = detector.detect(im)
    #         # print(boxes.shape)
    #         imDraw = ImageDraw.Draw(im)
    #         for box in boxes:
    #             x1 = int(box[0])
    #             y1 = int(box[1])
    #             x2 = int(box[2])
    #             y2 = int(box[3])
    #
    #             # print(box[4])
    #             imDraw.rectangle((x1, y1, x2, y2), outline='red',width=3)
    #         y = time.time()
    #         print(y - x)
    #         im.show()
            # im.save(r".\mtcnn结果\test5.02\1.jpg")

    # 检测一个文件夹
    # x = time.time()
    # with torch.no_grad() as grad:
    #     data_path = r".\MTCNN作业\MTCNN作业附件图片1"
    #     class_names = os.listdir(data_path)
    #     detector = Detector()
    #     for name in class_names:
    #         x = time.time()
    #         with Image.open(os.path.join(data_path,name))as im:
    #             w, h = im.size
    #             im.resize((int(w * 0.5), int(h * 0.5)))
    #
    #             boxes = detector.detect(im)
    #             # print(boxes.shape)
    #             imDraw = ImageDraw.Draw(im)
    #             for box in boxes:
    #                 x1 = int(box[0])
    #                 y1 = int(box[1])
    #                 x2 = int(box[2])
    #                 y2 = int(box[3])
    #
    #                 # print(box[4])
    #                 imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
    #             y = time.time()
    #             print(y - x)
    #             im.show()
                # im.save(r".\mtcnn结果\test5.03\{}".format(name))

    #检测视频

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture('1.mp4')
    # w = int(cap.get(3))
    # h = int(cap.get(4))
    # fps = cap.get(5)
    detector = Detector()
    count = 0
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
                    img.save(r"C:\Users\lieweiai\Desktop\wj\{}.jpg".format(count))
                    count += 1
                draw = ImageDraw.Draw(image)
                draw.rectangle((x1, y1,x2 ,int(y1+(y2-y1)*0.9)), outline='red', width=4)
                draw.text((x1 + 10, y1 + 10), text=f'{box[4]:.3f}', fill='red')
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

    # 裁剪脑袋
    # x = time.time()
    # with torch.no_grad() as grad:
    #     data_path = r"C:\Users\lieweiai\Desktop\新建文件夹 (3)\新建文件夹"
    #     class_names = os.listdir(data_path)
    #     detector = Detector()
    #     for name in class_names:
    #         x = time.time()
    #         with Image.open(os.path.join(data_path,name))as im:
    #             im = im.convert("RGB")
    #             boxes = detector.detect(im)
    #             # print(boxes.shape)
    #             # imDraw = ImageDraw.Draw(im)
    #             for box in boxes:
    #                 x1 = int(box[0])
    #                 y1 = int(box[1])
    #                 x2 = int(box[2])
    #                 y2 = int(box[3])
    #                 y2 = int(y1+(y2-y1)*0.9)
    #                 if (x2-x1)>100 and (y2-y1)>100:
    #                     img = im.crop((x1, y1, x2, y2))
    #                     img.save(r"C:\Users\lieweiai\Desktop\郭富城\{}".format(name))
    #                 else:
    #                     continue
    #             y = time.time()
    #             print(y - x)



