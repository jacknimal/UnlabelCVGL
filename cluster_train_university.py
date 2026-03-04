import argparse
import os.path as osp
import random
import re
import shutil
import numpy as np
import sys
import collections
import time
import datetime
from datetime import timedelta
import math

import logging
import json
import os

from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import cv2
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional

# --- 项目依赖 ---
from clustercontrast.evaluate.university import evaluate
from clustercontrast.models.objectives import InfoNCE
from sample4geo.loss.triplet_loss import TripletLoss
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.meters import AverageMeter

from clustercontrast.datasets.university_1652_drone import university_drone
from clustercontrast.datasets.university_1652_satellite import university_satellite
from clustercontrast.datasets.university_1652 import U1652DatasetEval, create_filtered_confidence_dataset_train, \
    get_transforms
from clustercontrast.trainners import get_test_loader, get_train_loader_satellite
from clustercontrast.evaluators import extract_features
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.models.cm import ClusterMemory
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

# ==============================================================================
# --- 动态日志目录配置 ---
# ==============================================================================
# 1. 获取当前时间戳作为文件夹名称 (格式: YYYY-MM-DD_HH-MM-SS)
current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 2. 定义存放路径: 当前主目录下的 outs/时间戳/
log_dir = os.path.join("outs", current_time_str)
# 3. 创建文件夹 (如果 outs 不存在也会一并创建)
os.makedirs(log_dir, exist_ok=True)
# 4. 定义日志文件路径
log_file_path = os.path.join(log_dir, "log.txt")

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # 将原来的 'app.log' 替换为动态生成的 log_file_path
    handlers=[logging.FileHandler(log_file_path, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 核心架构升级: DINOv2 + LoRA 微调模块
# ==============================================================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=8):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros((original_linear.in_features, r)))
        self.lora_B = nn.Parameter(torch.zeros((r, original_linear.out_features)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original_linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def inject_lora(model, r=8, alpha=8):
    for name, module in model.named_modules():
        if 'qkv' in name and isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, r, alpha))


class DINOv2_Geo(nn.Module):
    def __init__(self, pretrained_path, r=8):
        super().__init__()
        logger.info(f"Loading DINOv2 from {pretrained_path}...")
        self.backbone = torch.hub.load('./facebookresearch_dinov2_main/dinov2', 'dinov2_vitb14', source='local',
                                       pretrained=False)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        inject_lora(self.backbone, r=r, alpha=r)

        self.embed_dim = 768
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x1, x2=None):
        f1 = self.backbone(x1)
        if x2 is not None:
            f2 = self.backbone(x2)
            return f1, f2
        # 兼容原生项目 evalutors.py 中 outputs[-2] 的提取逻辑
        return (None, None, f1, None)


# ==============================================================================
# 第一阶段专用极简 Trainer (抛弃伪视图和对抗网络)
# ==============================================================================
class ClusterContrastTrainer_Stage1(object):
    def __init__(self, encoder, memory_satellite=None, memory_drone=None, device="cuda:0"):
        self.encoder = encoder
        self.memory_satellite = memory_satellite
        self.memory_drone = memory_drone
        self.device = device

    def train(self, epoch, data_loader_satellite, data_loader_drone, optimizer, print_freq=10, train_iters=400,
              logger=None):
        self.encoder.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i in range(train_iters):
            inputs_satellite = data_loader_satellite.next()
            inputs_drone = data_loader_drone.next()

            imgs_sat, pids_sat, _ = self._parse_data(inputs_satellite)
            imgs_dro, pids_dro, _ = self._parse_data(inputs_drone)

            # DINOv2 前向传播 (伪装成元组取[-2]以防报错)
            f_out_sat = self.encoder(imgs_sat)[-2]
            f_out_dro = self.encoder(imgs_dro)[-2]

            loss_satellite = self.memory_satellite(f_out_sat, pids_sat)
            loss_drone = self.memory_drone(f_out_dro, pids_dro)
            loss = loss_satellite + loss_drone

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.info(f'Epoch: [{epoch}][{i + 1}/{train_iters}]\t'
                            f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                            f'Loss Sat {loss_satellite:.3f}\tLoss Dro {loss_drone:.3f}')

    def _parse_data(self, inputs):
        imgs, _, pids, indexes = inputs
        return imgs.to(self.device), pids.to(self.device), indexes.to(self.device)


# ==============================================================================
# 第二阶段专用 Trainer (聚焦全局特征对比学习)
# ==============================================================================
def train_stage2(train_config, model, dataloader, loss_functions, optimizer, epoch, train_steps_per, scheduler=None,
                 scaler=None):
    model.train()
    losses = AverageMeter()
    time.sleep(0.1)
    optimizer.zero_grad(set_to_none=True)

    for query, reference, ids, labels in dataloader:
        if scaler:
            with autocast():
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)

                features1, features2 = model(query, reference)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1:
                    loss_infonce = loss_functions["infoNCE"](features1, features2, model.module.logit_scale.exp())
                else:
                    loss_infonce = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())

                # 【修复位置 1】: 拼接特征和标签
                features_concat = torch.cat([features1, features2], dim=0)
                labels_concat = torch.cat([labels, labels], dim=0)
                loss_triplet = loss_functions["Triplet"](features_concat, labels_concat)

                # DINOv2 为全局语义特征，无需复杂的区块重组损失(DSA)
                lossall = train_config.weight_infonce * loss_infonce + train_config.triplet_loss * loss_triplet

                losses.update(lossall.item())

            scaler.scale(lossall).backward()
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            labels = labels.to(train_config.device)
            features1, features2 = model(query, reference)

            loss_infonce = loss_functions["infoNCE"](features1, features2, model.logit_scale.exp())

            # 【修复位置 2】: 拼接特征和标签
            features_concat = torch.cat([features1, features2], dim=0)
            labels_concat = torch.cat([labels, labels], dim=0)
            loss_triplet = loss_functions["Triplet"](features_concat, labels_concat)

            lossall = train_config.weight_infonce * loss_infonce + train_config.triplet_loss * loss_triplet
            losses.update(lossall.item())
            lossall.backward()
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

    return losses.avg


# ==============================================================================
# 动态伪标签生成: 图过滤算法 (适配单视图特征提取)
# ==============================================================================
def graph_filtering(
        features_drone: torch.Tensor,
        pseudo_labels_drone: np.ndarray,
        uav_image_paths_list: List[str],
        features_satellite: torch.Tensor,
        satellite_image_paths_list: List[str],
        k_neighbors: int = 2,
        output_json_path: Optional[str] = None,
) -> List[Tuple[int, str, str, float]]:
    print("--- Starting Graph Filtering Process (DINOv2 Optimized) ---")

    features_drone = torch.nn.functional.normalize(features_drone, p=2, dim=1)
    features_satellite = torch.nn.functional.normalize(features_satellite, p=2, dim=1)

    sim_matrix = torch.matmul(features_drone, features_satellite.T)
    num_uav = features_drone.size(0)

    # 按照第一阶段获取的聚类 ID 分组
    num_clusters = pseudo_labels_drone.max() + 1
    uav_clusters = [[] for _ in range(num_clusters)]
    for uav_idx, label in enumerate(pseudo_labels_drone):
        if label != -1:
            uav_clusters[label].append(uav_idx)

    final_labeled_pairs = []

    for cluster_id, cluster_of_uav_indices in enumerate(uav_clusters):
        if not cluster_of_uav_indices:
            continue

        vote_scores = {}
        for uav_idx in cluster_of_uav_indices:
            _, nn_indices = torch.topk(sim_matrix[uav_idx], k=k_neighbors)
            for sat_idx in nn_indices:
                sat_idx_item = sat_idx.item()
                weight = sim_matrix[uav_idx, sat_idx_item].item()
                vote_scores[sat_idx_item] = vote_scores.get(sat_idx_item, 0) + weight

        if not vote_scores:
            continue

        target_satellite_idx = max(vote_scores, key=vote_scores.get)

        for uav_idx in cluster_of_uav_indices:
            uav_item_path = uav_image_paths_list[uav_idx]
            target_satellite_path = satellite_image_paths_list[target_satellite_idx]
            pair_confidence = sim_matrix[uav_idx, target_satellite_idx].item()
            final_labeled_pairs.append((target_satellite_idx, uav_item_path, target_satellite_path, pair_confidence))

    print(f"--- Process Finished. Generated {len(final_labeled_pairs)} high-quality training pairs. ---")

    if output_json_path and final_labeled_pairs:
        correct_matches = 0
        total_pairs = len(final_labeled_pairs)
        for pair_data in final_labeled_pairs:
            uav_path = pair_data[1]
            sat_path = pair_data[2]
            uav_id_match = re.search(r'/drone/(\d+)', uav_path)
            sat_id_match = re.search(r'/satellite/(\d+)', sat_path)
            if uav_id_match and sat_id_match and uav_id_match.group(1) == sat_id_match.group(1):
                correct_matches += 1

        accuracy = (correct_matches / total_pairs) * 100 if total_pairs > 0 else 0.0
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_entry = {
            "timestamp": current_time,
            "total_correct_matches": correct_matches,
            "total_pairs": total_pairs,
            "accuracy_percent": round(accuracy, 2)
        }
        all_records = []
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_records = []
        all_records.append(record_entry)
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"IOError: {e}")

    return final_labeled_pairs, {}


def log_cluster_statistics(cluster_labels, stage=""):
    cluster_count = collections.Counter(cluster_labels)
    logger.info(f"==> {stage} Clustering Statistics:")
    logger.info(f"    Number of clusters: {len(cluster_count)}")


# ==============================================================================
# MAIN FUNCTION CONTROLLER
# ==============================================================================
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # 无缝执行两阶段训练
    main_worker_stage1_intra_view(args)
    main_worker_stage2_inter_view(args)


def main_worker_stage1_intra_view(args):
    start_time = time.monotonic()
    iters = args.iters if (args.iters > 0) else None

    logger.info("==> Load unlabeled dataset")
    drone_dataset = university_drone(root=args.dataset, logger=logger)
    satellite_dataset = university_satellite(root=args.dataset, logger=logger)

    # 加载 DINOv2 及其 LoRA
    model_pretrained_path = osp.join(args.model_path, 'dinov2_vitb14_pretrain.pth')
    model = DINOv2_Geo(pretrained_path=model_pretrained_path, r=8)

    if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(args.device)

    # 仅优化具有梯度要求的参数 (LoRA)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  weight_decay=args.weight_decay)

    trainer = ClusterContrastTrainer_Stage1(encoder=model, device=args.device)
    best_score = 0

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DINOv2 特征紧凑，DBSCAN eps 建议维持在 0.75 上下
                cluster_d = DBSCAN(eps=0.75, min_samples=5, metric='precomputed', n_jobs=-1)

            logger.info('==> Extracting features for clustering...')
            cluster_loader_drone = get_test_loader(sorted(drone_dataset.dataset), height=args.height, width=args.width,
                                                   batch_size=args.batch_size, num_workers=args.num_workers)
            features_drone, _ = extract_features(model, cluster_loader_drone, print_freq=50, device=args.device,
                                                 logger=logger)
            features_drone = torch.cat([features_drone[f].unsqueeze(0) for f, _ in sorted(drone_dataset.dataset)], 0)
            del cluster_loader_drone

            cluster_loader_satellite = get_test_loader(sorted(satellite_dataset.dataset), height=args.height,
                                                       width=args.width, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)
            features_satellite, _ = extract_features(model, cluster_loader_satellite, print_freq=50, device=args.device,
                                                     logger=logger)
            features_satellite = torch.cat(
                [features_satellite[f].unsqueeze(0) for f, _ in sorted(satellite_dataset.dataset)], 0)
            del cluster_loader_satellite

            # 【完美优化】卫星视图无需聚类，每张图就是一个独立的类别 (0, 1, 2, ..., N-1)
            num_satellite_imgs = features_satellite.size(0)
            pseudo_labels_satellite = np.arange(num_satellite_imgs)

            # 无人机继续保留 DBSCAN
            rerank_dist_drone = compute_jaccard_distance(features_drone, k1=args.k1, k2=args.k2, search_option=3,
                                                         logger=logger)
            pseudo_labels_drone = cluster_d.fit_predict(rerank_dist_drone)

            log_cluster_statistics(pseudo_labels_satellite, stage="Satellite")
            log_cluster_statistics(pseudo_labels_drone, stage="Drone")
            del rerank_dist_drone

            num_cluster_s = len(set(pseudo_labels_satellite)) - (1 if -1 in pseudo_labels_satellite else 0)
            num_cluster_d = len(set(pseudo_labels_drone)) - (1 if -1 in pseudo_labels_drone else 0)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1: continue
                centers[labels[i]].append(features[i])
            centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
            return torch.stack(centers, dim=0)

        cluster_features_satellite = generate_cluster_features(pseudo_labels_satellite, features_satellite)
        cluster_features_drone = generate_cluster_features(pseudo_labels_drone, features_drone)
        del features_satellite, features_drone

        # Memory 初始化维度修正为 DINOv2 的 768 维
        memory_satellite = ClusterMemory(768, num_cluster_s, temp=args.temp, momentum=args.momentum,
                                         use_hard=args.use_hard).to(args.device)
        memory_drone = ClusterMemory(768, num_cluster_d, temp=args.temp, momentum=args.momentum,
                                     use_hard=args.use_hard).to(args.device)
        memory_satellite.features = F.normalize(cluster_features_satellite, dim=1).to(args.device)
        memory_drone.features = F.normalize(cluster_features_drone, dim=1).to(args.device)

        trainer.memory_satellite = memory_satellite
        trainer.memory_drone = memory_drone

        # pseudo_labels_satellite 已经是 numpy 数组，可以直接取 item() 进行匹配打包
        pseudo_labeled_dataset_satellite = [(fname, label.item()) for (fname, _), label in
                                            zip(sorted(satellite_dataset.dataset), pseudo_labels_satellite) if
                                            label != -1]
        pseudo_labeled_dataset_drone = [(fname, label.item()) for (fname, _), label in
                                        zip(sorted(drone_dataset.dataset), pseudo_labels_drone) if label != -1]

        data_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms((args.img_size, args.img_size),
                                                                                      mean=data_config["mean"],
                                                                                      std=data_config["std"])

        # Drone 统一使用标准的 get_train_loader_satellite 加载（无色彩迁移）
        train_loader_satellite = get_train_loader_satellite(args, satellite_dataset.dataset, args.height, args.width,
                                                            args.batch_size, args.num_workers, iters,
                                                            trainset=pseudo_labeled_dataset_satellite,
                                                            train_transformer=train_sat_transforms)
        train_loader_drone = get_train_loader_satellite(args, drone_dataset.dataset, args.height, args.width,
                                                        args.batch_size, args.num_workers, iters,
                                                        trainset=pseudo_labeled_dataset_drone,
                                                        train_transformer=train_drone_transforms)

        train_loader_satellite.new_epoch()
        train_loader_drone.new_epoch()

        trainer.train(epoch, train_loader_satellite, train_loader_drone, optimizer, print_freq=args.print_freq,
                      train_iters=iters, logger=logger)

        if epoch >= 0 and ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            test_batch = 64
            query_dataset_test = U1652DatasetEval(data_folder='/home/djx_yhl/University-Release/test/query_drone',
                                                  mode="query", transforms=val_transforms)
            query_dataloader_test = DataLoader(query_dataset_test, batch_size=test_batch, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True)
            gallery_dataset_test = U1652DatasetEval(
                data_folder='/home/djx_yhl/University-Release/test/gallery_satellite', mode="gallery",
                transforms=val_transforms, sample_ids=query_dataset_test.get_sample_ids())
            gallery_dataloader_test = DataLoader(gallery_dataset_test, batch_size=test_batch,
                                                 num_workers=args.num_workers, shuffle=False, pin_memory=True)

            r1_test = evaluate(config=args, model=model, query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, ranks=[1, 5, 10], step_size=1000, cleanup=True,
                               logger=logger)

            if r1_test > best_score:
                best_score = r1_test
                logger.info("New Best R@1 score in Stage 1: %f", best_score)
                save_path = '{}/model_best.pth'.format(args.logs_dir)
                if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)

    end_time = time.monotonic()
    print('Stage 1 total running time: ', timedelta(seconds=end_time - start_time))


def main_worker_stage2_inter_view(args):
    start_time = time.monotonic()
    args.epochs = 6
    args.checkpoint_start = osp.join(args.logs_dir, 'model_best.pth')

    logger.info("==> STAGE 2: Start cross-view matching")
    drone_dataset = university_drone(root=args.dataset, logger=logger)
    satellite_dataset = university_satellite(root=args.dataset, logger=logger)

    model_pretrained_path = osp.join(args.model_path, 'dinov2_vitb14_pretrain.pth')
    model = DINOv2_Geo(pretrained_path=model_pretrained_path, r=8)

    if args.checkpoint_start is not None and os.path.exists(args.checkpoint_start):
        logger.info(f"Loading Stage 1 weights from: {args.checkpoint_start}")
        model_state_dict = torch.load(args.checkpoint_start, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=True)

    if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(args.device)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_functions = {
        "infoNCE": InfoNCE(loss_function=loss_fn, device=args.device),
        "Triplet": TripletLoss(margin=args.triplet_loss)
    }

    scaler = GradScaler(init_scale=2. ** 10) if args.mixed_precision else None
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_steps = (len(drone_dataset.dataset) / args.batch_size) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_steps,
                                                num_warmup_steps=train_steps * args.warmup_epochs)

    best_score = 0
    data_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    for epoch in range(args.epochs):
        val_transforms, train_sat_transforms, train_drone_transforms = get_transforms((args.img_size, args.img_size),
                                                                                      mean=data_config["mean"],
                                                                                      std=data_config["std"])

        with torch.no_grad():
            if epoch == 0:
                cluster_d = DBSCAN(eps=0.75, min_samples=5, metric='precomputed', n_jobs=-1)

            cluster_loader_drone = get_test_loader(sorted(drone_dataset.dataset), height=args.height, width=args.width,
                                                   batch_size=args.batch_size, num_workers=args.num_workers)
            features_drone, _ = extract_features(model, cluster_loader_drone, print_freq=50, device=args.device,
                                                 logger=logger)
            features_drone = torch.cat([features_drone[f].unsqueeze(0) for f, _ in sorted(drone_dataset.dataset)], 0)
            del cluster_loader_drone

            cluster_loader_satellite = get_test_loader(sorted(satellite_dataset.dataset), height=args.height,
                                                       width=args.width, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)
            features_satellite, _ = extract_features(model, cluster_loader_satellite, print_freq=50, device=args.device,
                                                     logger=logger)
            features_satellite = torch.cat(
                [features_satellite[f].unsqueeze(0) for f, _ in sorted(satellite_dataset.dataset)], 0)
            del cluster_loader_satellite

            rerank_dist_drone = compute_jaccard_distance(features_drone, k1=args.k1, k2=args.k2, search_option=3,
                                                         logger=logger)
            pseudo_labels_drone = cluster_d.fit_predict(rerank_dist_drone)
            del rerank_dist_drone

        # 将生成的伪标签匹配结果也保存在当次的 outs/时间戳 目录下
        output_file = osp.join(log_dir, 'contrastive_image_pairs.json')
        uav_image_paths_list_main = [fname for fname, _ in sorted(drone_dataset.dataset)]
        satellite_image_paths_list_main = [fname for fname, _ in sorted(satellite_dataset.dataset)]

        results, _ = graph_filtering(
            features_drone=features_drone,
            pseudo_labels_drone=pseudo_labels_drone,
            uav_image_paths_list=uav_image_paths_list_main,
            features_satellite=features_satellite,
            satellite_image_paths_list=satellite_image_paths_list_main,
            k_neighbors=2,
            output_json_path=output_file,
        )

        threshold = 0.3 - epoch * 0.05
        print(f"Confidence threshold for epoch {epoch}: {threshold:.2f}")
        # 将 sample 权重也保存在同一个文件夹下，保持输出完全整洁
        sample_path = osp.join(log_dir, f'samples_epoch_{epoch}.pth')
        train_dataset = create_filtered_confidence_dataset_train(results_data=results, confidence_threshold=threshold,
                                                                 transforms_query=train_sat_transforms,
                                                                 transforms_gallery=train_drone_transforms,
                                                                 prob_flip=args.prob_flip,
                                                                 shuffle_batch_size=args.batch_size,
                                                                 output_samples_path=sample_path)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=not args.custom_sampling, pin_memory=True)
        if args.custom_sampling:
            train_dataloader.dataset.shuffle()

        train_loss = train_stage2(args, model, dataloader=train_dataloader, loss_functions=loss_functions,
                                  optimizer=optimizer, epoch=epoch, train_steps_per=len(train_dataloader),
                                  scheduler=scheduler, scaler=scaler)

        if epoch >= 0 and ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            test_batch = 64
            query_dataset_test = U1652DatasetEval(data_folder='/home/djx_yhl/University-Release/test/query_drone',
                                                  mode="query", transforms=val_transforms)
            query_dataloader_test = DataLoader(query_dataset_test, batch_size=test_batch, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True)
            gallery_dataset_test = U1652DatasetEval(
                data_folder='/home/djx_yhl/University-Release/test/gallery_satellite', mode="gallery",
                transforms=val_transforms, sample_ids=query_dataset_test.get_sample_ids())
            gallery_dataloader_test = DataLoader(gallery_dataset_test, batch_size=test_batch,
                                                 num_workers=args.num_workers, shuffle=False, pin_memory=True)

            r1_test = evaluate(config=args, model=model, query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, ranks=[1, 5, 10], step_size=1000, cleanup=True,
                               logger=logger)

            if r1_test > best_score:
                best_score = r1_test
                save_model_path = osp.join(args.model_path, f'dinov2_weights_e{epoch}_{r1_test:.4f}.pth')
                if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), save_model_path)
                else:
                    torch.save(model.state_dict(), save_model_path)

    end_time = time.monotonic()
    print('Stage 2 total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(
        description="Self-paced contrastive learning on unsupervised cross-view geo-localization")
    parser.add_argument('--model', default='dinov2_vitb14', type=str, help='backbone model')
    parser.add_argument('--handcraft_model', default=True, type=bool, help='use modified backbone')
    # 对于 DINOv2 vitb14，建议输入尺寸为 14 的倍数，392 (28x14) 最为契合
    parser.add_argument('--img_size', default=224, type=int, help='input image size')
    parser.add_argument('--views', default=2, type=int, help='only supports 2 branches retrieval')
    parser.add_argument('--record', default=True, type=bool)

    parser.add_argument('--nclasses', default=701, type=int)
    parser.add_argument('--triplet_loss', default=0.3, type=float)
    parser.add_argument('--resnet', default=False, type=bool)
    parser.add_argument('--weight_infonce', default=1.0, type=float)

    parser.add_argument('--mixed_precision', default=True, type=bool)
    parser.add_argument('--custom_sampling', default=True, type=bool)
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--gpu_ids', default=(0, 1, 2, 3), type=tuple)

    parser.add_argument('--clip_grad', default=100.0, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    # 学习率适配 DINOv2 LoRA 的常规参数
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--scheduler', default="cosine", type=str)
    parser.add_argument('--warmup_epochs', default=0.1, type=float)

    parser.add_argument('--dataset_name', default='U1652', type=str)
    parser.add_argument('--prob_flip', default=0.5, type=float)
    parser.add_argument('--model_path', default='./checkpoints', type=str)  # 指向你的权重文件夹

    parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)

    parser.add_argument('-d', '--dataset', type=str, default='/home/djx_yhl')
    parser.add_argument('--height', type=int, default=224, help="input height")
    parser.add_argument('--width', type=int, default=224, help="input width")

    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--momentum', type=float, default=0.2)
    parser.add_argument('--use-hard', action="store_true")

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--iters', type=int, default=600)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default="./checkpoints/logs")

    # 修复测试过程中的 AttributeError 报错
    parser.add_argument('--normalize_features', default=True, type=bool, help='normalize features in predict')

    main()
