import multiprocessing
import os
import traceback
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import random

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (loss0.data.item(), loss1.data.item(
    # ), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2netp'  # 'u2netp'

data_dir = "/content/"
tra_image_dir = os.path.join("fashion-dataset/images/")
tra_label_dir = os.path.join("processed_images_2/")

image_ext = '.jpg'
label_ext = '_mask.png'

model_dir = os.path.join('/content/drive/MyDrive/icloset', model_name + os.sep)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

epoch_num = 100000
batch_size_train = 76
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

fixed_tra_img_name_list = []
tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    label_name = os.path.join(data_dir, tra_label_dir, str(imidx) + label_ext)
    if not os.path.exists(label_name):
        print("label file not exists: ", label_name)
        continue
    fixed_tra_img_name_list.append(img_path)
    tra_lbl_name_list.append(label_name)

print("---")
print("train images: ", len(fixed_tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

# 划分训练集和验证集
combined = list(zip(fixed_tra_img_name_list, tra_lbl_name_list))
random.shuffle(combined)
split_idx = int(0.95 * len(combined))
train_set = combined[:split_idx]
val_set = combined[split_idx:]

train_img_list, train_lbl_list = zip(*train_set) if train_set else ([], [])
val_img_list, val_lbl_list = zip(*val_set) if val_set else ([], [])

train_num = len(train_img_list)
val_num = len(val_img_list)

# 训练集 DataLoader
salobj_dataset = SalObjDataset(
    img_name_list=list(train_img_list),
    lbl_name_list=list(train_lbl_list),
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True,
                               num_workers=multiprocessing.cpu_count(), pin_memory=True, persistent_workers=True)

# 验证集 DataLoader
val_salobj_dataset = SalObjDataset(
    img_name_list=list(val_img_list),
    lbl_name_list=list(val_lbl_list),
    transform=transforms.Compose([
        RescaleT(320),
        ToTensorLab(flag=0)]))
val_salobj_dataloader = DataLoader(val_salobj_dataset, batch_size=batch_size_val, shuffle=False,
                                   num_workers=0, pin_memory=True)

# ------- 3. define model --------
# define the net
if (model_name == 'u2net'):
    net = U2NET(3, 1)
elif (model_name == 'u2netp'):
    net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(
    0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")


def process():
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 100  # save the model every 100 iterations
    loss_log_path = os.path.join(model_dir, "loss_log.txt")
    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(
                    labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(
                d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                # 计算当前训练集 loss（一个 batch 的平均 loss）
                net.eval()
                train_loss = running_loss / ite_num4val if ite_num4val > 0 else 0
                train_tar_loss = running_tar_loss / ite_num4val if ite_num4val > 0 else 0

                # 计算一次验证 loss
                val_loss = 0.0
                val_tar_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_data in val_salobj_dataloader:
                        val_inputs, val_labels = val_data['image'], val_data['label']
                        val_inputs = val_inputs.type(torch.FloatTensor)
                        val_labels = val_labels.type(torch.FloatTensor)
                        if torch.cuda.is_available():
                            val_inputs_v, val_labels_v = Variable(val_inputs.cuda(), requires_grad=False), Variable(val_labels.cuda(), requires_grad=False)
                        else:
                            val_inputs_v, val_labels_v = Variable(val_inputs, requires_grad=False), Variable(val_labels, requires_grad=False)
                        d0, d1, d2, d3, d4, d5, d6 = net(val_inputs_v)
                        val_loss2, val_l = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, val_labels_v)
                        val_loss += val_l.data.item()
                        val_tar_loss += val_loss2.data.item()
                        val_count += 1
                        del d0, d1, d2, d3, d4, d5, d6, val_loss2, val_l
                avg_val_loss = val_loss / val_count if val_count > 0 else 0
                avg_val_tar_loss = val_tar_loss / val_count if val_count > 0 else 0

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" %
                           (ite_num, train_loss, train_tar_loss))
                print("Model saved." + model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" %
                      (ite_num, train_loss, train_tar_loss))
                # 记录 loss 到文件
                with open(loss_log_path, "a") as f:
                    f.write(f"save_point: ite:{ite_num}, train_loss:{train_loss:.6f}, train_tar_loss:{train_tar_loss:.6f}, val_loss:{avg_val_loss:.6f}, val_tar_loss:{avg_val_tar_loss:.6f}\n")
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0



if __name__ == "__main__":
    try:
        process()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        stacktrace = traceback.format_exc()
        print("An error occurred during training:")
        print(stacktrace)
    finally:
        # 训练完成后主动释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
