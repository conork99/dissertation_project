import logging
import sys
import numpy as np
import torch
import os
import argparse
from lib.unet_jcs import UNet
from torch.utils.data import DataLoader
from utils.dataloader import DR_grading_seg#, RandomRotFlip
from sklearn.metrics import roc_auc_score, jaccard_score
from scipy.spatial import distance
import cv2
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.metrics_MIL import calc_err_five_class

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='M:/LIFE703/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='unet_jcs_bce', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=4000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0,1', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.1,
                    help='balance factor to control edge and body loss')
parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float,
                        default=0.99, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,
                    default=200, help='every n epochs decay learning rate')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model_grading_seg/" + args.exp + "_{}_bs_beta_{}/".format(args.batch_size, args.beta)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
saved_model_path = os.path.join(snapshot_path, 'best_model_seg.pth')
# saved_model_path = os.path.join(snapshot_path, 'best_model_acc.pth')

save_se_pre_path = os.path.join(snapshot_path, 'pred/se_pre/')
if not os.path.exists(save_se_pre_path):
    os.makedirs(save_se_pre_path)
save_ex_pre_path = os.path.join(snapshot_path, 'pred/ex_pre/')
if not os.path.exists(save_ex_pre_path):
    os.makedirs(save_ex_pre_path)

save_ma_pre_path = os.path.join(snapshot_path, 'pred/ma_pre/')
if not os.path.exists(save_ma_pre_path):
    os.makedirs(save_ma_pre_path)

save_he_pre_path = os.path.join(snapshot_path, 'pred/he_pre/')
if not os.path.exists(save_he_pre_path):
    os.makedirs(save_he_pre_path)

save_total_pred_path = os.path.join(snapshot_path, 'pred/total_pre/')
if not os.path.exists(save_total_pred_path):
    os.makedirs(save_total_pred_path)

def calculate_metric_percase(pred, gt):

    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

    dice = dice_coef(gt, pred)
    mae = mean_absolute_error(gt, pred)
    # AUC = roc_auc_score(gt.flatten(), pred.flatten())
    # jc = jaccard_score(gt.flatten(), pred.flatten())
    return dice, mae# , jc, AUC
if __name__ == "__main__":
    model = UNet(n_classes=2)
    model = model.cuda()

    db_test = DR_grading_seg(base_dir=train_data_path, split='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    best_performance = 0.0
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    with torch.no_grad():

        total_metric = 0.0
        total_metric_se = 0
        total_metric_ex = 0
        total_metric_ma = 0
        total_metric_he = 0
        grade_total = np.zeros((len(testloader), 5))
        label_grade_total = np.zeros((len(testloader), 1))

        for i_batch, (sampled_batch, sampled_name) in enumerate(testloader):
            volume_batch, label_se, label_ex, label_ma, label_he, label_grade = sampled_batch['img'], \
                sampled_batch['SE_mask'], \
                sampled_batch['EX_mask'], sampled_batch['MA_mask'], sampled_batch['HE_mask'], sampled_batch[
                'grade_label']
            # edge_batch_com = edge_batch[:, 0, :, :] + edge_batch[:, 1, :, :]
            volume_batch, label_se, label_ex, label_ma, label_he, label_grade = volume_batch.float().cuda(), label_se.float().cuda(), \
                label_ex.float().cuda(), label_ma.float().cuda(), label_he.float().cuda(), label_grade.float().cuda()
            # volume_batch, label_batch, edge_batch = volume_batch.cuda(), label_batch.cuda(), edge_batch.cuda()

            lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, grade_pred = model(volume_batch)  # SE, EX, MA, HE, grading
            # mask = torch.sigmoid(ori_mask)

            # lateral_map_4 = mask[:, 0, ...]
            # lateral_map_3 = mask[:, 1, ...]
            # lateral_map_2 = mask[:, 2, ...]
            # lateral_map_1 = mask[:, 3, ...]

            grade_pred = F.softmax(grade_pred, dim=0)
            # print(grade_pred)
            lateral_map_4 = lateral_map_4.cpu().data.numpy().squeeze()
            lateral_map_3 = lateral_map_3.cpu().data.numpy().squeeze()
            lateral_map_2 = lateral_map_2.cpu().data.numpy().squeeze()
            lateral_map_1 = lateral_map_1.cpu().data.numpy().squeeze()
            grade_pred = grade_pred.cpu().data.numpy()

            label_se = label_se.cpu().data.numpy().squeeze()
            label_ex = label_ex.cpu().data.numpy().squeeze()
            label_ma = label_ma.cpu().data.numpy().squeeze()
            label_he = label_he.cpu().data.numpy().squeeze()
            label_grade = label_grade.cpu().data.numpy().squeeze()

            # lateral_map_4 = (lateral_map_4 > 0.5).astype(np.uint8)
            # lateral_map_3 = (lateral_map_3 > 0.5).astype(np.uint8)
            # lateral_map_2 = (lateral_map_2 > 0.5).astype(np.uint8)
            # lateral_map_1 = (lateral_map_1 > 0.5).astype(np.uint8)

            lateral_map_4 = np.argmax(lateral_map_4, axis=0)
            lateral_map_3 = np.argmax(lateral_map_3, axis=0)
            lateral_map_2 = np.argmax(lateral_map_2, axis=0)
            lateral_map_1 = np.argmax(lateral_map_1, axis=0)

            single_metric_se = calculate_metric_percase(lateral_map_4, label_se.astype(np.uint8))
            single_metric_ex = calculate_metric_percase(lateral_map_3, label_ex.astype(np.uint8))
            single_metric_ma = calculate_metric_percase(lateral_map_2, label_ma.astype(np.uint8))
            single_metric_he = calculate_metric_percase(lateral_map_1, label_he.astype(np.uint8))

            total_metric_se = total_metric_se + np.asarray(single_metric_se)
            total_metric_ex = total_metric_ex + np.asarray(single_metric_ex)
            total_metric_ma = total_metric_ma + np.asarray(single_metric_ma)
            total_metric_he = total_metric_he + np.asarray(single_metric_he)


            grade_total[i_batch * 1:i_batch * 1 + volume_batch.size(0), :] = grade_pred
            Y_hat = np.argmax(grade_total, axis=1)
            label_grade_total[i_batch * 1:i_batch * 1 + volume_batch.size(0), :] = label_grade

            img = volume_batch.cpu().data.numpy().squeeze()
            new_img = img.transpose(1,2,0)
            # print(img.shape)

            # Total_mask = lateral_map_4 *2 + lateral_map_3 *3 + lateral_map_2 *4 + lateral_map_1* 5
            # Total_mask_gt = label_se *2+ label_ex *3 + label_ma *4 + label_he*5

            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(1, 10, figsize=(12, 6))
            ax1.imshow(new_img, cmap='gray')
            ax2.imshow(lateral_map_4, cmap='gray')
            ax3.imshow(lateral_map_3, cmap='gray')
            ax4.imshow(lateral_map_2, cmap='gray')
            ax5.imshow(lateral_map_1, cmap='gray')
            ax6.imshow(new_img, cmap='gray')
            ax7.imshow(label_se, cmap='gray')
            ax8.imshow(label_ex, cmap='gray')
            ax9.imshow(label_ma, cmap='gray')
            ax10.imshow(label_he, cmap='gray')

            ax1.set_title('input')
            ax2.set_title('pred_se')
            ax3.set_title('pred_ex')
            ax4.set_title('pred_ma')
            ax5.set_title('pred_he')
            ax6.set_title('input')
            ax7.set_title('label_se')
            ax8.set_title('label_ex')
            ax9.set_title('label_ma')
            ax10.set_title('label_he')

            # plt.show()
            plt.axis('off')
            # plt.savefig(save_total_pred_path + sampled_name[0].replace('.h5', '.png'))
            plt.close()

        precision_score, spe, sen, f1_score, cm, auc = calc_err_five_class(Y_hat, label_grade_total, grade_total,
                                                                           if_test=True)

        avg_metric_se = (total_metric_se) / len(testloader)
        avg_metric_ex = (total_metric_ex) / len(testloader)
        avg_metric_ma = (total_metric_ma) / len(testloader)
        avg_metric_he = (total_metric_he) / len(testloader)


        print('SE: dice : %f, MAE : %f' % (avg_metric_se[0], avg_metric_se[1]))
        print('EX: dice : %f, MAE : %f' % (avg_metric_ex[0], avg_metric_ex[1]))
        print('MA: dice : %f, MAE : %f' % (avg_metric_ma[0], avg_metric_ma[1]))
        print('HE: dice : %f, MAE : %f' % (avg_metric_he[0], avg_metric_he[1]))
        # print(' f1 : %f, pre : %f, spe : %f, sen : %f, auc : %f' % (
        #  f1_score, precision_score, spe, sen, auc))











