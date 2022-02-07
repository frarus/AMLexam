import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import json
import skimage.color as color

class GTA5LAB(data.Dataset):
    def __init__(self, root, train=True, max_iters=None, crop_size=(1024, 512), mean=(104.00698793, 116.66876762, 122.67891434), scale=True, ignore_label=255):
        self.mean = mean
        self.crop_size = crop_size
        self.root = root
        self.list_path = osp.join(self.root, "train.txt")
        self.scale = scale
        self.ignore_label = ignore_label
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        #target che serve per FDA
        self.img_ids = [i_id.strip() for i_id in open('/content/drive/MyDrive/Datasets/Cityscapes/train.txt')]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        PATH = '/content/drive/MyDrive/Datasets/Cityscapes'
        self.filestarget = []
        for name in self.img_ids:

          names=name.split('/')[1].split('_')
          name = names[0]+'_'+names[1]+'_'+names[2]
          image_path = os.path.join (PATH,'images',name+'_leftImg8bit.png')
          self.filestarget.append({
                "image": image_path,
                "name": name
          })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        targetfiles=self.filestarget[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        targetimage=Image.open(targetfiles["image"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        targetimage=targetimage.resize(self.crop_size,Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        #LAB
        image = np.asarray(image)/255 #il /255 serve per sistemare i range, trovato online
        image_lab=color.rgb2lab(image)
        mean_s=np.mean(image_lab, axis=(0,1))
        std_s=np.std(image_lab, axis=(0,1))

        img_trg_lab=color.rgb2lab(targetimage)
        mean_t=np.mean(img_trg_lab, axis=(0,1))
        std_t=np.std(img_trg_lab, axis=(0,1))

        image_lab_transformed=((image_lab-mean_s)/std_s)*std_t+mean_t
        image=color.lab2rgb(image_lab_transformed)*255 #*255 sistema i range
        image = image.astype(np.uint8)     #sistema i range
        #im = Image.fromarray(image, "RGB")
        #im.save("/content/drive/MyDrive/test3/out"+str(index)+".jpeg")
        #END LAB

        image = np.asarray(image, np.float32)
        targetimage=np.asarray(targetimage, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            
