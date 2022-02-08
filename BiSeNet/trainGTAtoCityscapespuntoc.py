import argparse
import os
import torch.backends.cudnn as cudnn
import torch
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
from PIL import Image

from loss import DiceLoss
from model.discriminator_dsc_test import DiscriminatorDSC
from dataset.Cityscapes import Cityscapes
from dataset.Cityscapes_pseudo import Cityscapes_pseudo
from dataset.GTA5 import GTA5
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from SSL import create_pseudo_labels

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(optimizer, i_iter, lrate, num_steps, power):
    lr = lr_poly(lrate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = image.astype(int)

    final = np.zeros((x.shape[0], x.shape[1],3), np.ubyte)
    x[x==255]=19
    final[:,:]=colour_codes[x]
    #print(final.shape)
    return final

def val(args, model, dataloader):

    import os
    path = r"/content/drive/MyDrive/test"
    os.chdir(path)



    print('start val!')
    label_info = json.load(open('/content/drive/MyDrive/Datasets/Cityscapes/info.json', 'r'))
    
    label_info=label_info['palette']
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label, _, _) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            #print (predict.shape)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            img = colour_code_segmentation(predict, label_info)
            img = Image.fromarray(img, 'RGB')
            img.save("/content/drive/MyDrive/test/out"+str(i)+".jpeg")



            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())
            img = Image.fromarray(label, 'RGB')
            #img.save("/content/drive/MyDrive/test/label"+str(i)+".jpeg")


            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou

def main(params):
    parser = argparse.ArgumentParser()
    #args di cityscapes
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    #args di gta cityscapes
    # parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,         #da controllare
    #                     help="Path to the directory containing the source dataset.")
    # parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
    #                     help="Path to the file listing the images in the source dataset.")         #da controllare
    parser.add_argument("--ignore_label", type=int, default=255, help="The index of the label to ignore during the training.")
    """da qui in poi bene o male tutti da controllare"""
    parser.add_argument('--directory_source', type=str, default='/content/drive/MyDrive/Datasets/GTA5', help='directory for source data')
    parser.add_argument('--directory_target', type=str, default='/content/drive/MyDrive/Datasets/Cityscapes', help='directory for target data')
    parser.add_argument("--is_training", action="store_true", help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning_rate_D", type=float, default=1e-4, help="Base learning rate for discriminator.")
    parser.add_argument("--lambda_seg", type=float, default=0.1, help="lambda_seg.")
    parser.add_argument("--iter_size", type=int, default=125, help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--lambda_adv", type=float, default=0.001, help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--not_restore_last", action="store_true", help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num_steps", type=int, default=250000, help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=150000,  help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save_num_images", type=int, default=2, help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=5000,  help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot_dir", type=str, default='./snapshots/',  help="Where to save snapshots of the model.")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--set", type=str, default="train",  help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default="Vanilla", help="choose the GAN objective.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--update_pseudo_labels", type=str, default=1, help="How many epochs pseudo labels should be updated.")
    parser.add_argument("--save_dir_plabels", type=str, default=1, help="Directory where pseudo labels will be saved")


    args = parser.parse_args(params)

    #Inizio codice train gta cityscapes
    """Create the model and start the training."""

    cudnn.enabled = True 

    train_dataset_source = GTA5(args.directory_source)
    val_dataset = Cityscapes (train=False)

    # Define here your dataloaders --> target dataloader has been moved
    dataloader_source = DataLoader(train_dataset_source, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    dataloader_val = DataLoader(val_dataset, shuffle=False, num_workers=args.num_workers)

    # Create network

    # build model
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    #model.train()

    cudnn.benchmark = True

    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    scaler = amp.GradScaler()

    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

    # init D
    model_D = DiscriminatorDSC(num_classes=args.num_classes)

    model_D.train()
    model_D.cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()


    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    max_miou = 0

    for epoch in range(args.num_epochs):

        if epoch>0:
          train_dataset_target = Cityscapes_pseudo()
            
        else:
          train_dataset_target = Cityscapes ()
          
        dataloader_target = DataLoader(train_dataset_target, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
       

        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs) #learning rate for generator
        adjust_learning_rate_D(optimizer_D, epoch, args.learning_rate_D, args.num_epochs, args.power)
        tq = tqdm(total=(len(dataloader_target) + len (dataloader_source)) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []

        loss_D=0
        loss_segmentation=0
        loss_adversarial=0
        loss_seg_trg=0
        loss_D_trg_fake=0
        loss_seg_trg=0
        model.train()
        
        sourceloader_iter = enumerate(dataloader_source)
        targetloader_iter = enumerate(dataloader_target)

        for sub_i in range(args.iter_size):
            optimizer_D.zero_grad()
            optimizer.zero_grad()

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source

            _, batch = next(sourceloader_iter)
            images, labels, _, _ = batch
            data = images.cuda()
            label = labels.long().cuda()

            with amp.autocast():
                output_source, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output_source, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss_segmentation = loss1 + loss2 + loss3                    #LOSS SEGMENTATION
            
            scaler.scale(loss_segmentation).backward()                      #LOSS = LOSS_SEG_SRC
            loss_record.append(loss_segmentation.item())


            # train with target

            #IN CASE WITH PSEUDO-LABELS
            if epoch>0:                              #2 epoca, pseudolabels create
                _, batch = next(targetloader_iter)
                trg_img, trg_lbl, _, _ = batch
                trg_img = trg_img.cuda()
                trg_lbl = trg_lbl.long().cuda()
                with amp.autocast():
                    output_target, output_sup1, output_sup2 = model(trg_img)
                    loss1 = loss_func(output_target, label)
                    loss2 = loss_func(output_sup1, label)
                    loss3 = loss_func(output_sup2, label)
                    loss_seg_trg = loss1 + loss2 + loss3 
                
            
            #WITHOUT PSEUDOLABELS
            else:
                _, batch = next(targetloader_iter)
                images, _, _, _ = batch
                data = images.cuda()

                with amp.autocast():
                    output_target, _, _ = model(data)
                    loss_seg_trg=0
#---------------------------------------------------------------------------------------------------------------
            with amp.autocast():
                D_out=model_D(F.softmax(output_target))

            ######################################################################
            loss_D_trg_fake=bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
                                                                  #LOSS ADVERSARIAL
            #LOSS_D_TARGET_FAKE=LOSS_ADVERSARIAL

            loss_trg = args.lambda_adv * loss_D_trg_fake + loss_seg_trg
            scaler.scale(loss_trg).backward()               
            scaler.step(optimizer)

            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            output_source = output_source.detach()
            with amp.autocast():
                D_out = model_D(F.softmax(output_source))

            loss_D_source = bce_loss(D_out,
                              Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

            

            # train with target
            output_target = output_target.detach()
            with amp.autocast():
                D_out = model_D(F.softmax(output_target))

            loss_D_target = bce_loss(D_out,
                              Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())
            #DIVIDERE PER 2
            loss_D = loss_D_source/2+loss_D_target/2

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            tq.update(args.batch_size*2)

            scaler.update()
        loss_train_mean = np.mean(loss_record)
        tq.close() 
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('average loss for train : %f' % (loss_train_mean))

        torch.save({'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_segmentation,
                    'loss_record': loss_record},
                    os.path.join(args.save_model_path, 'latest_crossentropy_loss.pth'))

        #if epoch % args.validation_step == 0 and epoch != 0:
        precision, miou = val(args, model, dataloader_val)
        if miou > max_miou:
            max_miou = miou
            import os 
            os.makedirs(args.save_model_path, exist_ok=True)
            torch.save({'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_segmentation,
                    'loss_record': loss_record},
                    os.path.join(args.save_model_path, 'best_crossentropy_loss.pth'))
        writer.add_scalar('epoch/precision_val', precision, epoch)
        writer.add_scalar('epoch/miou val', miou, epoch)

        print(
        'iter = {0:8d}/{1:8d}, loss_segmentation = {2:.3f} loss_adversarial = {3:.3f} loss_D = {4:.3f} '.format(
            epoch, args.num_epochs, loss_segmentation, loss_D_trg_fake, loss_D))
        
        if ((epoch+1)%args.update_pseudo_labels==0):
            create_pseudo_labels(model, args.save_dir_plabels, args.num_workers, batch_size=1)



if __name__ == '__main__':
    params = [
        '--save_dir_plabels', '/content/drive/MyDrive/Datasets/pseudolabels',
        '--num_epochs', '50',
        '--learning_rate', '2.5e-2',
        '--data', './data/...',
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', '/content/drive/MyDrive/checkpoints_101_sgd_DSC_withpseudolabels',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        '--loss', 'crossentropy', #fine parametri train, ora metto quelli di train gta cityscapes

    ]
    main(params)