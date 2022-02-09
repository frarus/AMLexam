
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader

def create_pseudo_labels(model, save_dir, num_workers, batch_size):
   

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    model.eval()
    model.cuda()   
    train_dataset_target = Cityscapes ()
    targetloader = DataLoader(train_dataset_target, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    predicted_label = np.zeros((len(targetloader)*batch_size, 512, 1024))
    predicted_prob = np.zeros((len(targetloader)*batch_size, 512, 1024))
    image_name = []
    
    for index, batch in enumerate(targetloader):
        #if index % 100 == 0:
             #print( '%d processd' % index)
        image, _, _, name = batch
        output = model(Variable(image).cuda())
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])
  
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.5))])
    print (thres)
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print (thres)

    thres= 0.9
    for index in range(len(targetloader)*batch_size):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        label[(prob<thres)] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        #print (name)
        name = name + "_gtFine_labelIds.png"
        output.save('%s/%s' % (save_dir, name)) 
    
