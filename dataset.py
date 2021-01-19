from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import os

tf = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
])


class MyDataset(Dataset):

    def __init__(self, main_dir):

        self.dataset = []
        for face_dir in os.listdir(main_dir):
            # print(face_dir)
            for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
                self.dataset.append([os.path.join(main_dir, face_dir, face_filename), int(face_dir)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img = Image.open(data[0])
        img = img.convert("RGB")
        image_data = tf(img)
        label_data = data[1]
        return image_data, label_data


if __name__ == '__main__':
    mydataset = MyDataset(r"D:\程序代码\数据集\明星脸\民星脸split\train")
    # mydataset = MyDataset(r"data3")
    dataset = DataLoader(mydataset,5,shuffle=True)
    for data in dataset:
        print(data[0].shape)
        print(data[1].shape)
