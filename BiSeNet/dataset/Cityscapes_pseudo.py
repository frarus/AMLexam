import os
import numpy as np
from torch.utils import data
from PIL import Image
import json

def encode_segmap(mask, mapping, ignore_index):
    label_copy = ignore_index * np.ones(mask.shape, dtype=np.float32)
    for k, v in mapping:
        label_copy[mask == k] = v

    return label_copy


class Cityscapes_pseudo(data.Dataset):

    def __init__(self, crop_size=(1024,512), mean=(104.00698793, 116.66876762, 122.67891434), train=True, max_iters=None, ignore_index=255, ssl=None, train_mode=None):
        self.mean = mean
        self.crop_size = crop_size
        self.train = train
        self.set = 'train' if self.train else 'val'
        self.ignore_index = ignore_index
        self.files = []
        self.ssl = ssl
        self.train_mode = train_mode

        if self.train: 
            self.img_ids = [i_id.strip() for i_id in open('/content/drive/MyDrive/Datasets/Cityscapes/train.txt')]
        else:
            self.img_ids = [i_id.strip() for i_id in open('/content/drive/MyDrive/Datasets/Cityscapes/val.txt')]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.info = json.load(open('/content/drive/MyDrive/Datasets/Cityscapes/info.json', 'r'))
        self.class_mapping = self.info['label2train']
        
        PATH = '/content/drive/MyDrive/Datasets/Cityscapes'
        PATH_LBL = '/content/drive/MyDrive/Datasets/pseudolabels'

        for name in self.img_ids:

          names=name.split('/')[1].split('_')
          name = names[0]+'_'+names[1]+'_'+names[2]
          image_path = os.path.join (PATH,'images',name+'_leftImg8bit.png')
          
          label_path = os.path.join (PATH_LBL, name+'_gtFine_labelIds.png')
          #print (image_path)
          #print (label_path)
          #print (name)
          self.files.append({
                "image": image_path,
                "label": label_path,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        # open image and label file
        image = Image.open(file['image']).convert('RGB')
        label = Image.open(file['label'])
        name = file['name']

        # resize
        if "train" in self.set: 
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
        else:
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)

        # convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # remap the semantic label
        if not self.ssl:
            label = encode_segmap(label, self.class_mapping, self.ignore_index)

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
