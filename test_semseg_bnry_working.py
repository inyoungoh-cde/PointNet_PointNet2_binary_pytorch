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

# BASE_DIR/ROOT_DIR 설정: models 폴더를 패스에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 수정: S3DISDataLoader 대신, 우리의 ShapeNetDataLoader_forSS.py의 SemanticDataset 사용
from data_utils.ShapeNetDataLoader_forSS import SemanticDataset

# binary segmentation: 0 -> 'innerpoint', 1 -> 'boundarypoint'
seg_label_to_cat = {0: 'innerpoint', 1: 'boundarypoint'}

def parse_args():
    parser = argparse.ArgumentParser('Test Model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size in testing [default: 16]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: 0]')
    parser.add_argument('--num_point', type=int, default=2500, help='Number of points per sample [default: 2500]')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root (log directory)')
    parser.add_argument('--num_votes', type=int, default=3, help='Number of votes [default: 3]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
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

    # 데이터 경로: ShapeNet 형식의 custom dataset
    root = './binary_data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    # binary segmentation이므로, NUM_CLASSES는 (network 출력 채널은 1, 하지만 평가를 위해 두 클래스로 계산)
    NUM_CLASSES = 2
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size

    log_string(logger, "Start loading test data ...")
    TEST_DATASET = SemanticDataset(root=root, npoints=NUM_POINT, split='test', normal_channel=args.normal) #True
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=True)
    log_string(logger, "The number of test samples: %d" % len(TEST_DATASET))

    # MODEL LOADING
    # 여기서는 모델 이름은 args.model 대신, experiment_dir/logs 폴더 내 파일 이름을 사용
    model_files = os.listdir(experiment_dir / 'logs')
    if len(model_files) == 0:
        log_string(logger, "No model log file found!")
        exit(-1)
    model_name = model_files[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES, normal_channel=args.normal).cuda()  # network는 binary segmentation용으로 1 채널 출력 #
    checkpoint_path = str(experiment_dir / 'checkpoints' / 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    total_correct = 0
    total_seen = 0
    total_seen_class = [0, 0]
    total_correct_class = [0, 0]
    total_iou_deno_class = [0, 0]

    # 테스트 결과 저장을 위한 리스트 (각 파일별 결과 저장)
    test_results = {}

    with torch.no_grad():
        log_string(logger, '---- TEST EVALUATION ----')
        for batch_idx, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            # points: [B, npoints, channels] where channels=5, target: [B, npoints]
            points = points.float().cuda()  # [B, npoints, 5]
            target = target.long().cuda()     # [B, npoints]
            points = points.transpose(2, 1)    # [B, 5, npoints]


            # Forward pass
            seg_pred, _ = classifier(points)  # Expected shape: [B, npoints, 2]
            
            # Reshape prediction for evaluation
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)  # [B*N, 2]
            target = target.view(-1)  # [B*N]

            # Get predicted labels (argmax over class dimension)
            pred_label = seg_pred.argmax(dim=1).cpu().data.numpy()
            target_flat = target.cpu().data.numpy()

            # Accuracy calculation
            total_correct += np.sum(pred_label == target_flat)
            total_seen += pred_label.shape[0]

            # Compute per-class IoU
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(target_flat == l)
                total_correct_class[l] += np.sum((pred_label == l) & (target_flat == l))
                total_iou_deno_class[l] += np.sum((pred_label == l) | (target_flat == l))

            # Save per-sample results
            batch_points = points.transpose(2, 1).cpu().data.numpy()  # [B, npoints, 5]
            batch_xyz = batch_points[:, :, :3]  # Only x, y, z

            # Reshape predictions back to [B, npoints]
            pred_label_reshaped = pred_label.reshape(points.size(0), NUM_POINT)

            # Save test result for each sample in the batch
            for b in range(points.size(0)):
                pred_result = np.zeros((NUM_POINT, 4))  # [x, y, z, pred_label]
                pred_result[:, 0:3] = batch_xyz[b, :, :]
                pred_result[:, 3] = pred_label_reshaped[b, :]
                save_filename = predicted_results_dir / ('test_result_%03d.txt' % (batch_idx * BATCH_SIZE + b))
                np.savetxt(str(save_filename), pred_result, fmt='%.5f')
                
        # Compute overall metrics
        overall_acc = total_correct / float(total_seen)
        IoU_per_class = [total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6) for l in range(NUM_CLASSES)]
        mean_IoU = np.mean(IoU_per_class)
        avg_class_acc = np.mean([total_correct_class[l] / float(total_seen_class[l] + 1e-6) for l in range(NUM_CLASSES)])

        log_string(logger, 'Eval overall point accuracy: %.4f' % overall_acc)
        log_string(logger, 'Eval mean IoU: %.4f' % mean_IoU)
        log_string(logger, 'Eval average class accuracy: %.4f' % avg_class_acc)

        iou_per_class_str = '------- IoU per class --------\n'
        for l in range(NUM_CLASSES):
            iou = IoU_per_class[l]
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (seg_label_to_cat[l], iou)
        log_string(logger, iou_per_class_str)
        log_string(logger, 'Done!')

if __name__ == '__main__':
    args = parse_args()
    main(args)
