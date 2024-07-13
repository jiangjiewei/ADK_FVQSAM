"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import PIL
import cv2
from shutil import copyfile
import pickle
from PIL import Image
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import warnings
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    default='your path',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate moinrtdel on validation set')
parser.add_argument('--pretrained', default='pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fine-tuning',default='True', action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu',type=int,
                    help='GPU id to use.')  #default=3,
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def fundus_test():
    args = parser.parse_args()

    dataset_subdir = [
                      'test'
                      ]
    model_names = ['densenet121',
                   ]
    for index_name in range(0,len(model_names)):
        args.arch = model_names[index_name]
        for i in range(0,len(dataset_subdir)):
            dataset_dir = args.data
            resultset_dir = os.path.join("result",model_names[index_name])
            args, model, val_transforms = load_modle_trained(args)
            mk_result_dir(args, resultset_dir)
            fundus_test_exec(args, model, val_transforms, dataset_dir,resultset_dir)

def load_modle_trained(args):
    model = models.densenet121(pretrained=True)
    model = new_clsnet(model)
    print(model)
    model = nn.DataParallel(model).cuda()
    normalize = transforms.Normalize(mean=[0.5765036, 0.34929818, 0.2401832],
                                        std=[0.2179051, 0.19200659, 0.17808074])
    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    print("=> loading checkpoint###")
    if args.arch.find('alexnet') != -1:
        pre_name = './alexnet'
    elif args.arch.find('inception_v3') != -1:
        pre_name = './inception_v3'
    elif args.arch.find('densenet121') != -1:
        pre_name = './densenet121'
    elif args.arch.find('resnet50') != -1:
        pre_name = './resnet50'
    else:
        print('### please check the args.arch###')
        exit(-1)

    PATH = pre_name + '_model_best.pth.tar'
    # PATH = './densenet_nodatanocost_best.ckpt'

    # PATH = pre_name + '_checkpoint.pth.tar'
    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint['state_dict'])
    if args.arch.find('alexnet') != -1:
        model = torch.nn.DataParallel(model).cuda()  # for models trained by single GPU: Alexnet
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    print('best_epoch and best_acc1 is: ', start_epoch, best_acc1)
    return args, model, val_transforms

def mk_result_dir(args,testdata_dir='./data/val1'):
    testdatadir = testdata_dir
    model_name = args.arch
    result_dir = testdatadir + '/' + model_name
    grade1_grade2 = 'normal_keratitis'
    grade1_grade3 = 'normal_other'
    grade2_grade3 = 'keratitis_other'
    grade2_grade1 = 'keratitis_normal'
    grade3_grade1 = 'other_normal'
    grade3_grade2 = 'other_keratitis'

    # grade1_grade1 = 'normal_normal'
    # grade2_grade2 = 'keratitis_keratitis'
    # grade3_grade3 = 'other_other'

    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
        os.makedirs(result_dir + '/' + grade1_grade2)
        os.makedirs(result_dir + '/' + grade1_grade3)
        os.makedirs(result_dir + '/' + grade2_grade3)
        os.makedirs(result_dir + '/' + grade2_grade1)
        os.makedirs(result_dir + '/' + grade3_grade1)
        os.makedirs(result_dir + '/' + grade3_grade2)

        # os.makedirs(result_dir + '/' + grade1_grade1)
        # os.makedirs(result_dir + '/' + grade2_grade2)
        # os.makedirs(result_dir + '/' + grade3_grade3)

def fundus_test_exec(args,model,val_transforms,testdata_dir='./data/val1',resultset_dir='./data/val1'):
    # switch to evaluate mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdatadir = testdata_dir
    desdatadir = resultset_dir + '/' + args.arch
    model.eval()
    str_end = ('.jpg', '.JPG', '.bmp', '.jpeg','.JPEG','.BMP','.tif','.TIF','.png','.PNG')
    grade1_grade2 = 'normal_keratitis'
    grade1_grade3 = 'normal_other'
    grade2_grade3 = 'keratitis_other'
    grade2_grade1 = 'keratitis_normal'
    grade3_grade1 = 'other_normal'
    grade3_grade2 = 'other_keratitis'

    grade1_grade1 = 'normal_normal'
    grade2_grade2 = 'keratitis_keratitis'
    grade3_grade3 = 'other_other'

    with torch.no_grad():
        grade1_num = 0
        grade1_grade1_num = 0
        grade1_grade2_num = 0
        grade1_grade3_num = 0
        list_grade1_grade2=[grade1_grade2]
        list_grade1_grade3=[grade1_grade3]
        grade1_2=[grade1_grade2]
        grade1_3=[grade1_grade3]
        grade1_1=[grade1_grade1]
        root = testdatadir + '/normal'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)
            # img_PIL_Tensor = img_PIL_Tensor.cuda(args.gpu, non_blocking=True)
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]

            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]


            grade1_num = grade1_num + 1
            print(grade1_num)
            if pred_0 == 1:
                # print('ok to ok')
                grade1_grade1_num = grade1_grade1_num + 1
                grade1_1.append(prob_list)
            elif pred_0 == 0:
                # print('ok to location')
                grade1_grade2_num = grade1_grade2_num + 1
                list_grade1_grade2.append(img)
                grade1_2.append(prob_list)
                file_new_1 = desdatadir + '/normal_keratitis' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality to quality')
                grade1_grade3_num = grade1_grade3_num + 1
                list_grade1_grade3.append(img)
                grade1_3.append(prob_list)
                file_new_1 = desdatadir + '/normal_other' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade1_grade1_num, grade1_grade2_num, grade1_grade3_num)

        grade2_grade1_num = 0
        grade2_grade3_num = 0
        grade2_grade2_num = 0
        grade2_num = 0
        list_grade2_grade1=[grade2_grade1]
        list_grade2_grade3=[grade2_grade3]
        grade2_1=[grade2_grade1]
        grade2_3=[grade2_grade3]
        grade2_2=[grade2_grade2]
        root = testdatadir + '/keratitis'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            # img_PIL.show()  # 原始图片
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            # image = cv2.imread(os.path.join(root, img))  # image = image.unsqueeze(0) # PIL_image = Image.fromarray(image)
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]

            grade2_num = grade2_num + 1
            if pred_0 == 1:
                # print('location to ok')
                grade2_grade1_num = grade2_grade1_num + 1
                list_grade2_grade1.append(img)
                grade2_1.append(prob_list)
                file_new_1 = desdatadir + '/keratitis_normal' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                # print('location to quality')
                grade2_grade3_num = grade2_grade3_num + 1
                list_grade2_grade3.append(img)
                grade2_3.append(prob_list)
                file_new_1 = desdatadir + '/keratitis_other' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('location to location')
                grade2_grade2_num = grade2_grade2_num + 1
                grade2_2.append(prob_list)
        print(grade2_grade1_num, grade2_grade2_num, grade2_grade3_num)

        grade3_grade1_num = 0
        grade3_grade3_num = 0
        grade3_grade2_num = 0
        grade3_num = 0
        list_grade3_grade1=[grade3_grade1]
        list_grade3_grade2=[grade3_grade2]
        grade3_1=[grade3_grade1]
        grade3_2=[grade3_grade2]
        grade3_3=[grade3_grade3]
        root = testdatadir + '/other'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]

            grade3_num = grade3_num + 1
            if pred_0 == 1:
                # print('quality to ok')
                grade3_grade1_num = grade3_grade1_num + 1
                list_grade3_grade1.append(img)
                grade3_1.append(prob_list)
                file_new_1 = desdatadir + '/other_normal' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 0:
                # print('quality  to location')
                grade3_grade2_num = grade3_grade2_num + 1
                list_grade3_grade2.append(img)
                grade3_2.append(prob_list)
                file_new_1 = desdatadir + '/other_keratitis' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality  to quality')
                grade3_grade3_num = grade3_grade3_num + 1
                grade3_3.append(prob_list)
        print(grade3_grade1_num, grade3_grade2_num, grade3_grade3_num)


    confusion_matrix = [ [grade1_grade1_num, grade1_grade2_num, grade1_grade3_num],
                         [grade2_grade1_num, grade2_grade2_num, grade2_grade3_num],
                         [grade3_grade1_num, grade3_grade2_num, grade3_grade3_num]]
    print('confusion_matrix:')
    print (confusion_matrix)

    result_confusion_file = args.arch + '_1.txt'
    result_pro_file =  args.arch + '_2.txt'
    result_value_bin = args.arch + '_3.txt'


    with open(desdatadir + '/' + result_confusion_file, "w") as file_object:
        for i in confusion_matrix:
            file_object.writelines(str(i) + '\n')
        file_object.writelines('ERROR_images\n')
        for i in list_grade1_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade3:
            file_object.writelines(str(i) + '\n')

        for i in list_grade2_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade3:
            file_object.writelines(str(i) + '\n')

        for i in list_grade3_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade2:
            file_object.writelines(str(i) + '\n')
        file_object.close()

    with open(desdatadir + '/' + result_pro_file, "w") as file_object:
        for i in grade1_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade2_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade3_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        file_object.close()

    with open(desdatadir + '/' + result_value_bin, "wb") as file_object:
        pickle.dump(confusion_matrix, file_object)  # 顺序存入变量
        pickle.dump(grade1_1, file_object)
        pickle.dump(grade1_2, file_object)
        pickle.dump(grade1_3, file_object)
        pickle.dump(grade2_1, file_object)
        pickle.dump(grade2_2, file_object)
        pickle.dump(grade2_3, file_object)
        pickle.dump(grade3_1, file_object)
        pickle.dump(grade3_2, file_object)
        pickle.dump(grade3_3, file_object)
        file_object.close()




def fundus_test_exec_external(args,model,val_transforms,testdata_dir='./data/val1',resultset_dir='./data/val1'):
    # switch to evaluate mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdatadir = testdata_dir
    desdatadir = resultset_dir + '/' + args.arch
    model.eval()
    str_end = ('.jpg', '.JPG', '.bmp', '.jpeg','.JPEG','.BMP','.tif','.TIF','.png','.PNG')
    grade1_grade2 = 'normal_keratitis'
    grade1_grade3 = 'normal_other'
    grade2_grade3 = 'keratitis_other'
    grade2_grade1 = 'keratitis_normal'
    grade3_grade1 = 'other_normal'
    grade3_grade2 = 'other_keratitis'

    grade1_grade1 = 'normal_normal'
    grade2_grade2 = 'keratitis_keratitis'
    grade3_grade3 = 'other_other'

    with torch.no_grad():
        grade1_num = 0
        grade1_grade1_num = 0
        grade1_grade2_num = 0
        grade1_grade3_num = 0
        list_grade1_grade2=[grade1_grade2]
        list_grade1_grade3=[grade1_grade3]
        grade1_2=[grade1_grade2]
        grade1_3=[grade1_grade3]
        grade1_1=[grade1_grade1]
        # root = testdatadir + '/normal'
        root = testdatadir
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)
            # img_PIL_Tensor = img_PIL_Tensor.cuda(args.gpu, non_blocking=True)
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list =  prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]

            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 =  pred[0]


            grade1_num = grade1_num + 1
            print(grade1_num)
            if pred_0 == 1:
                # print('ok to ok')
                grade1_grade1_num = grade1_grade1_num + 1
                grade1_1.append(prob_list)
                file_new_1 = desdatadir + '/normal_normal' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 0:
                # print('ok to location')
                grade1_grade2_num = grade1_grade2_num + 1
                list_grade1_grade2.append(img)
                grade1_2.append(prob_list)
                file_new_1 = desdatadir + '/normal_keratitis' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality to quality')
                grade1_grade3_num = grade1_grade3_num + 1
                list_grade1_grade3.append(img)
                grade1_3.append(prob_list)
                file_new_1 = desdatadir + '/normal_other' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        print(grade1_grade1_num, grade1_grade2_num, grade1_grade3_num)


if __name__ == '__main__':
    fundus_test()
