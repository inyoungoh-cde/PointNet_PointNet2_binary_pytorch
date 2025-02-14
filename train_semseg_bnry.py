"""
Author: Benny
Date: Nov 2019 (modified for binary semantic segmentation)
"""
import argparse
import os
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import torch
import provider

from data_utils.ShapeNetDataLoader_forSS import SemanticDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_label_to_cat = {0: 'class0', 1: 'class1'}

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg_binary', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=2500, help='Point Number [default: 2500]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    return parser.parse_args()

def main(args):
    def log_string(message):
        logger.info(message)
        print(message)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # CREATE DIR
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = './binary_data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    NUM_CLASSES = 1  # binary segmentation: 0 or 1
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    log_string("Start loading training data ...")
    TRAIN_DATASET = SemanticDataset(root=root, npoints=NUM_POINT, split='train', normal_channel=True)
    log_string("Start loading test data ...")
    TEST_DATASET = SemanticDataset(root=root, npoints=NUM_POINT, split='test', normal_channel=True)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    # weights (if needed) -- binary segmentation might not need class weights;
    weights = None

    log_string("The number of training samples is: %d" % len(TRAIN_DATASET))
    log_string("The number of test samples is: %d" % len(TEST_DATASET))

    # MODEL LOADING
    print(args.model, "<-------look at")
    MODEL = importlib.import_module(args.model)
    #shutil.copy(f'models/{args.model}.py', str(experiment_dir))
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate: %f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda m: bn_momentum_adjust(m, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            # Data augmentation: rotate xyz around z-axis
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)  # [B, C, N], C=5

            seg_pred, trans_feat = classifier(points)  
            # seg_pred shape: [B, N, 1]; flatten to [B * N]
            seg_pred = seg_pred.view(-1)
            # target: flatten to [B * N] and convert to float
            target_flat = target.view(-1).float()

            loss = criterion(seg_pred, target_flat, trans_feat, weights)
            loss.backward()
            optimizer.step()

            # Threshold at 0.5 for binary prediction
            pred_choice = (seg_pred.cpu().data.numpy() > 0.5).astype(np.int32)
            batch_label = target_flat.cpu().data.numpy().astype(np.int32)
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Saving model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        # Evaluation
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            # For binary segmentation, IoU can be computed for class 0 and 1 separately
            total_seen_class = [0, 0]
            total_correct_class = [0, 0]
            total_iou_deno_class = [0, 0]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.view(-1)
                target_flat = target.view(-1).float()

                loss = criterion(seg_pred, target_flat, trans_feat, weights)
                loss_sum += loss.item()

                pred_choice = (seg_pred.cpu().data.numpy() > 0.5).astype(np.int32)
                batch_label = target_flat.cpu().data.numpy().astype(np.int32)
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)

                # Compute per-class statistics for binary segmentation (classes: 0 and 1)
                for l in [0, 1]:
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))

            mIoU = np.mean([total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6) for l in [0, 1]])
            log_string('Eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('Eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('Eval mean IoU: %f' % mIoU)
            log_string('Eval point avg class acc: %f' % (
                np.mean([total_correct_class[l] / float(total_seen_class[l] + 1e-6) for l in [0, 1]])))

            iou_per_class_str = '------- IoU --------\n'
            for l in [0, 1]:
                iou = total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6)
                iou_per_class_str += 'class %s, IoU: %.3f \n' % (seg_label_to_cat[l], iou)
            log_string(iou_per_class_str)
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Saving best model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Best model saved.')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)
