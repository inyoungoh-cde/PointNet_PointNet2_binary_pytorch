"""
Author: Benny
Date: Nov 2019 (Modified for binary semantic segmentation on ShapeNet custom dataset)
"""
import argparse
import os
import logging
import datetime
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from data_utils.ShapeNetDataLoader_forSS import SemanticDataset

seg_label_to_cat = {0: 'class0', 1: 'class1'}

def parse_args():
    parser = argparse.ArgumentParser('Test Model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size in testing [default: 16]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: 0]')
    parser.add_argument('--num_point', type=int, default=2500, help='Number of points per sample [default: 2500]')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root (log directory)')
    parser.add_argument('--num_votes', type=int, default=3, help='Number of votes [default: 3]')
    return parser.parse_args()

def log_string(logger, message):
    logger.info(message)
    print(message)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = Path('./log/sem_seg') / args.log_dir
    experiment_dir.mkdir(exist_ok=True, parents=True)
    predicted_results_dir = experiment_dir / 'predicted_results'
    predicted_results_dir.mkdir(exist_ok=True)

    # LOG setting
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = experiment_dir / 'eval.txt'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)

    root = './binary_data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    NUM_CLASSES = 2
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size

    log_string(logger, "Start loading test data ...")
    TEST_DATASET = SemanticDataset(root=root, npoints=NUM_POINT, split='test', normal_channel=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=True)
    log_string(logger, "The number of test samples: %d" % len(TEST_DATASET))

    model_files = os.listdir(experiment_dir / 'logs')
    if len(model_files) == 0:
        log_string(logger, "No model log file found!")
        exit(-1)
    model_name = model_files[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(1).cuda()
    checkpoint_path = str(experiment_dir / 'checkpoints' / 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    total_correct = 0
    total_seen = 0
    total_seen_class = [0, 0]
    total_correct_class = [0, 0]
    total_iou_deno_class = [0, 0]

    test_results = {}

    with torch.no_grad():
        log_string(logger, '---- TEST EVALUATION ----')
        for batch_idx, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points = points.float().cuda()
            target = target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, _ = classifier(points)
            seg_pred = seg_pred.view(-1)
            pred_label = (seg_pred.cpu().data.numpy() > 0.5).astype(np.int32)
            target_flat = target.view(-1).cpu().data.numpy().astype(np.int32)

            total_correct += np.sum(pred_label == target_flat)
            total_seen += pred_label.shape[0]

            for l in [0, 1]:
                total_seen_class[l] += np.sum(target_flat == l)
                total_correct_class[l] += np.sum((pred_label == l) & (target_flat == l))
                total_iou_deno_class[l] += np.sum(((pred_label == l) | (target_flat == l)))

            batch_points = points.transpose(2, 1).cpu().data.numpy()
            batch_xyz = batch_points[:, : , :3]  # x, y, z


            pred_label_reshaped = pred_label.reshape(points.size(0), NUM_POINT)

            for b in range(points.size(0)):
                pred_result = np.zeros((NUM_POINT, 4))
                pred_result[:, 0:3] = batch_xyz[b, :, :]
                pred_result[:, 3] = pred_label_reshaped[b, :]
                save_filename = predicted_results_dir / ('test_result_%03d.txt' % (batch_idx * BATCH_SIZE + b))
                np.savetxt(str(save_filename), pred_result, fmt='%.5f')
        
        overall_acc = total_correct / float(total_seen)
        IoU_per_class = [total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6) for l in [0, 1]]
        mean_IoU = np.mean(IoU_per_class)
        avg_class_acc = np.mean([total_correct_class[l] / float(total_seen_class[l] + 1e-6) for l in [0, 1]])

        log_string(logger, 'Eval overall point accuracy: %.4f' % overall_acc)
        log_string(logger, 'Eval mean IoU: %.4f' % mean_IoU)
        log_string(logger, 'Eval average class accuracy: %.4f' % avg_class_acc)
        iou_per_class_str = '------- IoU per class --------\n'
        for l in [0, 1]:
            iou = IoU_per_class[l]
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (seg_label_to_cat[l], iou)
        log_string(logger, iou_per_class_str)
        log_string(logger, 'Done!')

if __name__ == '__main__':
    args = parse_args()
    main(args)
