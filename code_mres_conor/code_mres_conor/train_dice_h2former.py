import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import logging
import time
import random
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import os
import argparse
from lib.pranet_h2former import PraNet
from utils.utils import clip_gradient
from utils.dataloader import DR_grading_seg  #, RandomRotFlip
from utils.criterion import BinaryDiceLoss, CriterionCrossEntropyEdgeParsing_boundary_attention_loss
from torch.nn import BCEWithLogitsLoss, MSELoss, BCELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, jaccard_score
from torch.nn import functional as F
from sklearn import metrics
from utils.metrics_MIL import calc_err_five_class

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='M:/LIFE703/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='DR_grading_seg_dice_h2former', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.1,
                    help='balance factor to control edge and body loss')
parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float,
                        default=0.5, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,
                    default=200, help='every n epochs decay learning rate')

args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = "../model_grading_seg/" + args.exp + "_{}_bs_beta_{}".format(args.batch_size, args.beta)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

"""reproducible"""

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

"""reproducible"""

# num_classes = 2
#patch_size = (128, 128, 80)


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
    # AUC = roc_auc_score(gt.flatten(), pred.flatten())
    # jc = jaccard_score(gt.flatten(), pred.flatten())
    return dice# , jc
def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]

    # fpr = float(np.logical_and(pred == 1, neq).sum())/(real==0).sum()
    # fnr = float(np.logical_and(pred == 0, neq).sum())/(real==1).sum()

    precision_score = metrics.precision_score(real, pred, average='macro')
    recall_score = metrics.recall_score(real, pred, average='macro')
    f1_score = metrics.f1_score(real, pred, average='macro')
    confusion_matrix = metrics.confusion_matrix(real, pred)
    return err, precision_score, recall_score, f1_score, confusion_matrix

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = PraNet()
    model = model.cuda()

    db_train = DR_grading_seg(base_dir=train_data_path,
                    split='train'
                       )
    db_val = DR_grading_seg(base_dir=train_data_path,
                   split='val')

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=4,
                             pin_memory=True, worker_init_fn=worker_init_fn, shuffle=True)
    testloader = DataLoader(db_val, batch_size=1, shuffle=False)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    bce_loss = BCEWithLogitsLoss()
    # mse_loss = MSELoss()
    ce_loss = CrossEntropyLoss()
    dice_loss = BinaryDiceLoss()

    # criterion = CriterionCrossEntropyEdgeParsing_boundary_attention_loss(loss_weight=[1, 1, 4])
    # criterion.cuda()

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader) + 1
    lr_ = base_lr
    best_acc = 0.0
    best_seg = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_se, label_ex, label_ma, label_he, label_grade  = sampled_batch['img'], sampled_batch['SE_mask'], \
                sampled_batch['EX_mask'], sampled_batch['MA_mask'], sampled_batch['HE_mask'], sampled_batch['grade_label']
            #edge_batch_com = edge_batch[:, 0, :, :] + edge_batch[:, 1, :, :]
            volume_batch, label_se ,label_ex, label_ma, label_he, label_grade = volume_batch.float().cuda(), label_se.float().cuda(), \
                label_ex.float().cuda(), label_ma.float().cuda(), label_he.float().cuda(), label_grade.long().cuda()
            #volume_batch, label_batch, edge_batch = volume_batch.cuda(), label_batch.cuda(), edge_batch.cuda()

            lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, grade_pred = model(volume_batch) # SE, EX, MA, HE, grading
            # ori_mask, grade_pred = model(volume_batch)



            # mask = torch.sigmoid(ori_mask)
            # lateral_map_4 = mask[:, 0, ...]
            # lateral_map_3 = mask[:, 1, ...]
            # lateral_map_2 = mask[:, 2, ...]
            # lateral_map_1 = mask[:, 3, ...]



            SE_loss = dice_loss(lateral_map_4.squeeze(), label_se.squeeze().float())
            EX_loss = dice_loss(lateral_map_3.squeeze(), label_ex.squeeze().float())
            MA_loss = dice_loss(lateral_map_2.squeeze(), label_ma.squeeze().float())
            HE_loss = dice_loss(lateral_map_1.squeeze(), label_he.squeeze().float())
            grade_loss = ce_loss(grade_pred.squeeze(), label_grade.squeeze().long())


            #edge_loss = edge_cup_loss + edge_disc_loss
            loss = SE_loss + EX_loss + MA_loss+ HE_loss + args.beta * (grade_loss)
            #loss = criterion(preds, [label_batch, edge_batch])

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/se_loss', SE_loss, iter_num)
            writer.add_scalar('loss/ex_loss', EX_loss, iter_num)
            writer.add_scalar('loss/ma_loss', MA_loss, iter_num)
            writer.add_scalar('loss/he_loss', HE_loss, iter_num)
            writer.add_scalar('loss/grade_loss', grade_loss, iter_num)


            logging.info(
                'iteration %d : loss : %f, se_loss: %f, ex_loss: %f, ma_loss: %f, he_loss: %f, grade_loss: %f' %  #, , edge_disc_loss: %f
                (iter_num, loss.item(), SE_loss.item(), EX_loss.item(),  MA_loss.item(), HE_loss.item(), grade_loss.item()))  # edge_cup_loss.item(), edge_disc_loss.item()


            #  save every 1000 item_num
            # if iter_num % 1000 == 0:
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        # change lr
        if epoch_num % args.decay_epoch == 0:
            lr_ = base_lr * args.decay_rate ** (epoch_num // 100)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

        if  epoch_num % 10 == 0: # epoch_num > 0 and
            model.eval()
            with torch.no_grad():
                total_metric = 0.0
                grade_total = np.zeros((len(testloader), 5))
                label_grade_total = np.zeros((len(testloader), 1))

                for i_batch, (sampled_batch, sample_name) in enumerate(testloader):
                    volume_batch, label_se, label_ex, label_ma, label_he, label_grade = sampled_batch['img'], \
                    sampled_batch['SE_mask'], \
                        sampled_batch['EX_mask'], sampled_batch['MA_mask'], sampled_batch['HE_mask'], sampled_batch[
                        'grade_label']
                    # edge_batch_com = edge_batch[:, 0, :, :] + edge_batch[:, 1, :, :]
                    volume_batch, label_se, label_ex, label_ma, label_he, label_grade = volume_batch.float().cuda(), label_se.float().cuda(), \
                        label_ex.float().cuda(), label_ma.float().cuda(), label_he.float().cuda(), label_grade.float().cuda()
                    # volume_batch, label_batch, edge_batch = volume_batch.cuda(), label_batch.cuda(), edge_batch.cuda()

                    # ori_mask, grade_pred = model(volume_batch)  # SE, EX, MA, HE, grading
                    lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, grade_pred = model(
                        volume_batch)  # SE, EX, MA, HE, grading
                    # mask = torch.sigmoid(ori_mask)
                    # lateral_map_4 = mask[:, 0, ...]
                    # lateral_map_3 = mask[:, 1, ...]
                    # lateral_map_2 = mask[:, 2, ...]
                    # lateral_map_1 = mask[:, 3, ...]

                    grade_pred = F.softmax(grade_pred,dim=0)
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

                    lateral_map_4 = (lateral_map_4 > 0.5).astype(np.uint8)
                    lateral_map_3 = (lateral_map_3 > 0.5).astype(np.uint8)
                    lateral_map_2 = (lateral_map_2 > 0.5).astype(np.uint8)
                    lateral_map_1 = (lateral_map_1 > 0.5).astype(np.uint8)

                    single_metric_se = calculate_metric_percase(lateral_map_4, label_se.astype(np.uint8))
                    single_metric_ex= calculate_metric_percase(lateral_map_3, label_ex.astype(np.uint8))
                    single_metric_ma = calculate_metric_percase(lateral_map_2, label_ma.astype(np.uint8))
                    single_metric_he = calculate_metric_percase(lateral_map_1, label_he.astype(np.uint8))
                    total_metric = total_metric + np.asarray(single_metric_se) + np.asarray(single_metric_ex) + np.asarray(single_metric_ma) + np.asarray(single_metric_he)

                    grade_total[i_batch * 1:i_batch * 1 + volume_batch.size(0), :] = grade_pred
                    Y_hat = np.argmax(grade_total, axis=1)
                    label_grade_total[i_batch * 1:i_batch * 1 + volume_batch.size(0), :] = label_grade


                precision_score, spe, sen, f1_score, cm, auc = calc_err_five_class(Y_hat, label_grade_total, grade_total, if_test=False)



                avg_metric = (total_metric) / len(testloader) / 4


                if avg_metric > best_seg:
                    best_seg = avg_metric
                    save_mode_path = os.path.join(snapshot_path, 'best_model_seg.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    logging.info('best seg performance dice score:{}'.format(best_seg))

                logging.info('iteration %d : dice : %f' % (iter_num, avg_metric))
                writer.add_scalar('val/avg_dice', avg_metric, iter_num)
                # writer.add_scalar('val/avg_jc', avg_metric[1], iter_num)
                # writer.add_scalar('val/avg_AUC', avg_metric[2], iter_num)

                if f1_score > best_acc:
                    best_acc = f1_score
                    save_mode_path = os.path.join(snapshot_path, 'best_model_acc.pth')
                    torch.save(model.state_dict(), save_mode_path)

                    logging.info("save model to {}".format(save_mode_path))
                    logging.info('best grade performance f1 score: {}'.format(best_acc) )
                writer.add_scalar('val/avg_f1', f1_score, iter_num)
                writer.add_scalar('val/avg_pre', precision_score, iter_num)
                writer.add_scalar('val/avg_sen', sen, iter_num)
                writer.add_scalar('val/avg_spe', spe, iter_num)
                writer.add_scalar('val/avg_auc', auc, iter_num)
                logging.info('iteration %d : f1 : %f, pre : %f, spe : %f, sen : %f, auc : %f' % (iter_num, f1_score, precision_score, spe, sen, auc))


                model.train()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()