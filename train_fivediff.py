import cv2
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import joblib

# --------------------
# 配置参数
# --------------------
class Config:
    # 数据参数
    data_root = "./data/fivediff"
    label_csv = "./data/fivediff_labels.csv"
    image_size = 512
    test_size = 0.15
    valid_size = 0.15
    # 模型参数
    backbone = "resnet101"
    pretrained = True
    dropout_rate = 0.3
    # 训练参数
    batch_size = 32
    lr = 1e-5
    epochs = 100
    weight_decay = 2e-3
    warmup_epochs = 10
    warmup_init_lr = 1e-7
    # 增强参数
    mixup_alpha = 0.3
    cutmix_alpha = 0.2
    # 保存路径
    save_dir = "./saved_fivediffmodels"
    log_dir = "./training_fivedifflogs"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 只保留面积标签配置
    area_label = 'spalling_area'      # 剥落面积的列名
    # 单任务，权重为 1.0
    area_loss_weight = 1.0
    # 添加服役寿命标签配置
    cycle_label = 'remaining_cycles'  # 服役寿命的列名
    cycle_loss_weight = 1.0           # 服役寿命损失的权重

# --------------------
# 数据准备模块
# --------------------
class YSZDataset(Dataset):
    def __init__(self, df, transform=None, mode='train', area_scaler=None, cycle_scaler=None):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.area_scaler = area_scaler
        self.cycle_scaler = cycle_scaler
        self.data_root = Config.data_root

        if mode != 'test':
            # 处理剥落面积标签
            if self.area_scaler is not None:
                self.area_labels = self.area_scaler.transform(
                    df[Config.area_label].values.reshape(-1, 1)
                ).flatten()
            else:
                self.area_scaler = StandardScaler()
                self.area_labels = self.area_scaler.fit_transform(
                    df[Config.area_label].values.reshape(-1, 1)
                ).flatten()

            # 处理服役寿命标签
            if Config.cycle_label in df.columns:
                if self.cycle_scaler is not None:
                    self.cycle_labels = self.cycle_scaler.transform(
                        df[Config.cycle_label].values.reshape(-1, 1)
                    ).flatten()
                else:
                    self.cycle_scaler = StandardScaler()
                    self.cycle_labels = self.cycle_scaler.fit_transform(
                        df[Config.cycle_label].values.reshape(-1, 1)
                    ).flatten()
            else:
                print(f"Warning: Column '{Config.cycle_label}' not found in CSV. Using zeros for cycle labels.")
                self.cycle_labels = np.zeros(len(df), dtype=np.float32)
                if self.cycle_scaler is None:
                    self.cycle_scaler = StandardScaler()
                    self.cycle_scaler.fit(np.zeros((len(df),1)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.df.iloc[idx]['image_name'])
        # 使用 OpenCV 读取，然后转 PIL 处理灰度图，兼容 albumentations
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or failed to load: {img_path}")

        if self.mode == 'test':
            # 对于测试模式，两个标签都设为0
            area_label_tensor = torch.tensor(0, dtype=torch.float32)
            cycle_label_tensor = torch.tensor(0, dtype=torch.float32)
        else:
            area_label_tensor = torch.tensor(self.area_labels[idx], dtype=torch.float32)
            # 获取寿命标签，如果不存在则使用0 (已在__init__中处理)
            cycle_val = self.cycle_labels[idx] if hasattr(self, 'cycle_labels') else 0.0
            cycle_label_tensor = torch.tensor(cycle_val, dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        # 返回两个标签
        return image, (area_label_tensor, cycle_label_tensor)


# --------------------
# 增强策略
# --------------------
def get_train_augmentations():
    return A.Compose([
        A.RandomResizedCrop(
            size=(Config.image_size, Config.image_size),
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.4
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5)
        ], p=0.4),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        A.OneOf([
            A.CLAHE(p=1),
            A.Equalize(p=1),
            A.Sharpen(p=1)
        ], p=0.4),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ], p=1.0)


def get_valid_augmentations():
    return A.Compose([
        A.Resize(
            height=Config.image_size,
            width=Config.image_size,
            interpolation=1
                 ),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ],p=1.0)


# --------------------
# 混合增强策略
# --------------------
def mixup_data(x, y_tuple, alpha=1.0): # y_tuple 是 (y_area, y_cycle)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device) # 确保 index 在同一设备

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_area, y_cycle = y_tuple # 解包标签

    y_area_a, y_area_b = y_area, y_area[index]
    y_cycle_a, y_cycle_b = y_cycle, y_cycle[index]

    return mixed_x, (y_area_a, y_cycle_a), (y_area_b, y_cycle_b), lam


# --------------------
# 模型架构
# --------------------
class EnhancedYSZModel(nn.Module):
    def __init__(self):   
        super().__init__()
        # 特征提取 (使用 ResNet101)
        weights = ResNet101_Weights.IMAGENET1K_V1 if Config.pretrained else None
        base = resnet101(weights=weights)
        original_first_conv = base.conv1
        self.features = nn.Sequential(
            nn.Conv2d(1, original_first_conv.out_channels,
                     kernel_size=original_first_conv.kernel_size,
                     stride=original_first_conv.stride,
                     padding=original_first_conv.padding,
                     bias=original_first_conv.bias),
            *list(base.children())[1:-2]
        )

        with torch.no_grad():
            new_weight = original_first_conv.weight.mean(dim=1, keepdim=True)
            self.features[0].weight.copy_(new_weight)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet101 特征维度是 2048
        backbone_output_features = 2048

        # 创建新的统一 area_head，直接接收 Backbone 输出
        self.area_head = nn.Sequential(
            nn.Linear(backbone_output_features, 1024), # 2048 -> 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),                         # Dropout
            nn.Linear(1024, 512),                    # 1024 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),                         # Dropout
            nn.Linear(512, 256),                     # 512 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),                         # Dropout
            nn.Linear(256, 128),                     # 256 -> 128 (输出层)
            nn.ReLU(),
            nn.Dropout(0.1),                         # Dropout
            nn.Linear(128, 1),                       # 128 -> 1 (输出层)
        )

        # 创建新的 cycle_head
        self.cycle_head = nn.Sequential(
            nn.Linear(backbone_output_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        self._init_weights()
        self._freeze_backbone()
        self.to(Config.device)

    def _init_weights(self):
        # 只初始化新的 area_head
        for m in self.area_head.modules(): # 直接遍历 area_head
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 初始化新的 cycle_head
        for m in self.cycle_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_backbone(self):
        total_layers = len(list(self.features.children()))
        freeze_idx = 0
        print(f"Freezing the first {freeze_idx} layers ({len(list(self.features))} total) of the ResNet101 backbone (Full Fine-tuning).")
        for idx, child in enumerate(self.features.children()):
            if idx < freeze_idx:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        features = self.features(x)
        pooled = self.adaptive_pool(features).squeeze(-1).squeeze(-1)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)

        # 直接将 pooled (2048维) 送入 area_head
        area_pred = self.area_head(pooled).squeeze(-1)
        # 将 pooled 送入 cycle_head
        cycle_pred = self.cycle_head(pooled).squeeze(-1)

        if area_pred.dim() == 0:
            area_pred = area_pred.unsqueeze(0)
        if cycle_pred.dim() == 0:
            cycle_pred = cycle_pred.unsqueeze(0)
            
        return area_pred, cycle_pred # 返回两个预测值


# --------------------
# 训练引擎
# --------------------
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = float('inf')
        # 只记录 area loss
        self.logs = {
            'train_area_loss': [],
            'valid_area_loss': [],
            'train_cycle_loss': [],
            'valid_cycle_loss': [],
            'train_total_loss': [],
            'valid_total_loss': []
        }

    def train_epoch(self, loader):
        self.model.train()
        running_area_loss = 0.0
        running_cycle_loss = 0.0
        running_total_loss = 0.0

        for images, labels_tuple in tqdm(loader, desc="Training"):
            images = images.to(Config.device)
            area_labels, cycle_labels = labels_tuple[0].to(Config.device), labels_tuple[1].to(Config.device)

            # Mixup 处理两个标签
            images, (targets_area_a, targets_cycle_a), (targets_area_b, targets_cycle_b), lam = mixup_data(
                images, (area_labels, cycle_labels), Config.mixup_alpha
            )

            self.optimizer.zero_grad()
            area_preds, cycle_preds = self.model(images)

            # 计算两个任务的损失
            loss_area = lam * self.criterion(area_preds, targets_area_a) + \
                        (1 - lam) * self.criterion(area_preds, targets_area_b)
            loss_cycle = lam * self.criterion(cycle_preds, targets_cycle_a) + \
                         (1 - lam) * self.criterion(cycle_preds, targets_cycle_b)
            
            # 加权总损失
            total_loss = (Config.area_loss_weight * loss_area) + \
                         (Config.cycle_loss_weight * loss_cycle)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = images.size(0)
            running_area_loss += loss_area.item() * batch_size
            running_cycle_loss += loss_cycle.item() * batch_size
            running_total_loss += total_loss.item() * batch_size

        dataset_size = len(loader.dataset)
        if dataset_size == 0:
             return 0.0, 0.0, 0.0
        return (
            running_area_loss / dataset_size,
            running_cycle_loss / dataset_size,
            running_total_loss / dataset_size
        )

    def valid_epoch(self, loader):
        self.model.eval()
        running_area_loss = 0.0
        running_cycle_loss = 0.0
        running_total_loss = 0.0

        with torch.no_grad():
            for images, labels_tuple in tqdm(loader, desc="Validation"):
                images = images.to(Config.device)
                area_labels, cycle_labels = labels_tuple[0].to(Config.device), labels_tuple[1].to(Config.device)

                area_preds, cycle_preds = self.model(images)
                loss_area = self.criterion(area_preds, area_labels)
                loss_cycle = self.criterion(cycle_preds, cycle_labels)

                total_loss = (Config.area_loss_weight * loss_area) + \
                             (Config.cycle_loss_weight * loss_cycle)

                batch_size = images.size(0)
                running_area_loss += loss_area.item() * batch_size
                running_cycle_loss += loss_cycle.item() * batch_size
                running_total_loss += total_loss.item() * batch_size

        dataset_size = len(loader.dataset)
        if dataset_size == 0:
            return 0.0, 0.0, 0.0
        return (
            running_area_loss / dataset_size,
            running_cycle_loss / dataset_size,
            running_total_loss / dataset_size
        )

    def run(self, train_loader, valid_loader):
        for epoch in range(Config.epochs):
            # --- 学习率预热逻辑 --- START ---
            if epoch < Config.warmup_epochs:
                # 计算当前预热阶段的学习率 (线性增加)
                warmup_factor = (epoch + 1) / Config.warmup_epochs
                current_lr = Config.warmup_init_lr + (Config.lr - Config.warmup_init_lr) * warmup_factor
                # 手动设置优化器的学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                # 预热结束后，如果当前 epoch 是 warmup_epochs (预热刚结束的第一个周期)，
                # 需要确保优化器的学习率是配置的基础学习率 Config.lr，
                # 因为 CosineAnnealingLR 会从优化器当前的 lr 开始衰减。
                # 虽然 CosineAnnealingLR 理论上从初始 lr 衰减，但为保险起见在此设置。
                if epoch == Config.warmup_epochs:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = Config.lr
            # --- 学习率预热逻辑 --- END ---

            train_area_loss, train_cycle_loss, train_total_loss = self.train_epoch(train_loader)
            valid_area_loss, valid_cycle_loss, valid_total_loss = self.valid_epoch(valid_loader)

            self.logs['train_area_loss'].append(train_area_loss)
            self.logs['valid_area_loss'].append(valid_area_loss)
            self.logs['train_cycle_loss'].append(train_cycle_loss)
            self.logs['valid_cycle_loss'].append(valid_cycle_loss)
            self.logs['train_total_loss'].append(train_total_loss)
            self.logs['valid_total_loss'].append(valid_total_loss)

            # 调度器步进 (预热期间也步进，但实际 LR 被手动覆盖)
            self.scheduler.step()

            # 保存最佳模型基于验证集 area loss
            if valid_total_loss < self.best_loss:
                self.best_loss = valid_total_loss
                torch.save(self.model.state_dict(),
                           os.path.join(Config.save_dir, 'best_model.pth'))

            # 打印日志
            print(f"Epoch {epoch + 1}/{Config.epochs}")
            print(f"Train Area Loss: {train_area_loss:.4f} | Train Cycle Loss: {train_cycle_loss:.4f} | Train Total Loss: {train_total_loss:.4f}")
            print(f"Valid Area Loss: {valid_area_loss:.4f} | Valid Cycle Loss: {valid_cycle_loss:.4f} | Valid Total Loss: {valid_total_loss:.4f}")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.8f}")
            print("-" * 50)

        self._save_logs()

    def _save_logs(self):
        os.makedirs(Config.log_dir, exist_ok=True)
        log_path = os.path.join(Config.log_dir, 'training_log_multitask.json')
        with open(log_path, 'w') as f:
            json.dump(self.logs, f)

    def evaluate(self, loader, area_scaler, cycle_scaler, output_csv_path=None):
        self.model.eval()
        all_area_preds, all_cycle_preds = [], []
        all_area_labels, all_cycle_labels = [], []
        image_names_for_csv = []
        original_indices_for_csv = []

        with torch.no_grad():
            for i, (images, labels_tuple) in tqdm(enumerate(loader), desc="Evaluating", total=len(loader)):
                images = images.to(Config.device)
                area_preds, cycle_preds = self.model(images)

                all_area_preds.extend(area_preds.cpu().numpy())
                all_cycle_preds.extend(cycle_preds.cpu().numpy())
                all_area_labels.extend(labels_tuple[0].cpu().numpy())
                all_cycle_labels.extend(labels_tuple[1].cpu().numpy())
                
                # 为了保存CSV，我们需要图像名。假设loader.dataset.df存在且包含图像名
                # 注意：这要求原始的Dataset的df在评估时仍然可访问，并且顺序一致
                # 如果使用了shuffle=True的DataLoader，这里的对应关系会打乱，需要额外处理
                # 简单起见，我们假设评估时的DataLoader没有shuffle，或者df的顺序是固定的
                start_idx = i * loader.batch_size
                end_idx = start_idx + len(images) # 处理最后一个不完整的batch
                if hasattr(loader.dataset, 'df') and 'image_name' in loader.dataset.df.columns:
                    image_names_for_csv.extend(loader.dataset.df['image_name'].iloc[start_idx:end_idx].values)
                if hasattr(loader.dataset, 'df') and 'original_index' in loader.dataset.df.columns:
                     original_indices_for_csv.extend(loader.dataset.df['original_index'].iloc[start_idx:end_idx].values)


        all_area_preds = np.array(all_area_preds)
        all_cycle_preds = np.array(all_cycle_preds)
        all_area_labels = np.array(all_area_labels)
        all_cycle_labels = np.array(all_cycle_labels)

        # 反向转换
        area_preds_orig = area_scaler.inverse_transform(all_area_preds.reshape(-1, 1)).flatten()
        area_labels_orig = area_scaler.inverse_transform(all_area_labels.reshape(-1, 1)).flatten()
        cycle_preds_orig = cycle_scaler.inverse_transform(all_cycle_preds.reshape(-1, 1)).flatten()
        cycle_labels_orig = cycle_scaler.inverse_transform(all_cycle_labels.reshape(-1, 1)).flatten()

        # 绘制散点图
        plot_prediction_scatter(area_preds_orig, area_labels_orig, 'area')
        plot_prediction_scatter(cycle_preds_orig, cycle_labels_orig, 'cycle')

        # 保存预测结果到CSV
        if output_csv_path:
            results_df_data = {
                Config.area_label + '_pred': area_preds_orig,
                Config.area_label + '_true': area_labels_orig,
                Config.cycle_label + '_pred': cycle_preds_orig,
                Config.cycle_label + '_true': cycle_labels_orig
            }
            if image_names_for_csv and len(image_names_for_csv) == len(area_preds_orig):
                results_df_data['image_name'] = image_names_for_csv
            if original_indices_for_csv and len(original_indices_for_csv) == len(area_preds_orig):
                results_df_data['original_index'] = original_indices_for_csv
            
            results_df = pd.DataFrame(results_df_data)
            # 将image_name和original_index（如果存在）放到前面
            cols = list(results_df.columns)
            if 'original_index' in cols:
                cols.insert(0, cols.pop(cols.index('original_index')))
            if 'image_name' in cols:
                cols.insert(0, cols.pop(cols.index('image_name')))
            results_df = results_df[cols]
            results_df.to_csv(output_csv_path, index=False)
            print(f"Predictions saved to {output_csv_path}")


        # 返回两个任务的指标
        area_metrics = {
            'mae': np.mean(np.abs(area_preds_orig - area_labels_orig)),
            'rmse': np.sqrt(np.mean((area_preds_orig - area_labels_orig) ** 2)),
            'r2': 1 - np.sum((area_preds_orig - area_labels_orig) ** 2) / \
                  np.sum((area_labels_orig - np.mean(area_labels_orig)) ** 2)
        }
        cycle_metrics = {
            'mae': np.mean(np.abs(cycle_preds_orig - cycle_labels_orig)),
            'rmse': np.sqrt(np.mean((cycle_preds_orig - cycle_labels_orig) ** 2)),
            'r2': 1 - np.sum((cycle_preds_orig - cycle_labels_orig) ** 2) / \
                  np.sum((cycle_labels_orig - np.mean(cycle_labels_orig)) ** 2)
        }
        return {'area': area_metrics, 'cycle': cycle_metrics}


# --------------------
# 主程序
# --------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(Config.save_dir, exist_ok=True)

    df = pd.read_csv(Config.label_csv)
    train_df, test_df = train_test_split(df, test_size=Config.test_size, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=Config.valid_size, random_state=42)

    # 创建数据集
    train_dataset = YSZDataset(train_df, get_train_augmentations(), mode='train')
    valid_dataset = YSZDataset(
        valid_df,
        get_valid_augmentations(),
        mode='valid',
        area_scaler=train_dataset.area_scaler,
        cycle_scaler=train_dataset.cycle_scaler
    )
    test_dataset = YSZDataset(
        test_df,
        get_valid_augmentations(),
        mode='test',
        area_scaler=train_dataset.area_scaler,
        cycle_scaler=train_dataset.cycle_scaler
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size * 2, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size * 2)

    # 初始化模型
    model = EnhancedYSZModel().to(device)
    criterion = nn.HuberLoss(delta=1.5)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 修改学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30%的时间用于预热
        div_factor=25,  # 初始学习率 = max_lr/25
        final_div_factor=1e4  # 最终学习率 = max_lr/10000
    )

    # 开始训练
    trainer = Trainer(model, criterion, optimizer, scheduler)
    trainer.run(train_loader, valid_loader)

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(Config.save_dir, 'best_model.pth')))

    print("\nEvaluating final model performance on Validation Set...")
    predictions_csv_path = os.path.join(Config.log_dir, "validation_predictions.csv")
    val_metrics = trainer.evaluate(valid_loader, train_dataset.area_scaler, train_dataset.cycle_scaler, output_csv_path=predictions_csv_path)
    
    print("\nFinal Validation Area Prediction Metrics:")
    print(f"MAE: {val_metrics['area']['mae']:.4f}")
    print(f"RMSE: {val_metrics['area']['rmse']:.4f}")
    print(f"R²: {val_metrics['area']['r2']:.4f}")
    print("\nFinal Validation Cycle Prediction Metrics:")
    print(f"MAE: {val_metrics['cycle']['mae']:.4f}")
    print(f"RMSE: {val_metrics['cycle']['rmse']:.4f}")
    print(f"R²: {val_metrics['cycle']['r2']:.4f}")

    # 保存 scaler
    joblib.dump(train_dataset.area_scaler, os.path.join(Config.save_dir, "scaler_area.pkl"))
    joblib.dump(train_dataset.cycle_scaler, os.path.join(Config.save_dir, "scaler_cycle.pkl"))

    # 绘制训练曲线
    log_path = os.path.join(Config.log_dir, 'training_log_multitask.json')
    plot_training_curves(log_path)
    print(f"\n训练曲线已保存至: {Config.log_dir}")

# --------------------
# 可视化工具
# --------------------
def plot_training_curves(log_path):
    with open(log_path) as f:
        logs = json.load(f)

    plt.figure(figsize=(10, 6))

    # 只绘制 area loss
    plt.plot(logs['train_area_loss'], label='Training Area Loss', color='blue', alpha=0.7)
    plt.plot(logs['valid_area_loss'], label='Validation Area Loss', color='cyan', alpha=0.7)
    plt.plot(logs['train_cycle_loss'], label='Training Cycle Loss', color='green', alpha=0.7)
    plt.plot(logs['valid_cycle_loss'], label='Validation Cycle Loss', color='lime', alpha=0.7)
    plt.plot(logs['train_total_loss'], label='Training Total Loss', color='red', linestyle='--', alpha=0.8)
    plt.plot(logs['valid_total_loss'], label='Validation Total Loss', color='magenta', linestyle='--', alpha=0.8)
    plt.title('Multi-task Prediction Training Progress', pad=10, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    curves_path = os.path.join(Config.log_dir, 'training_curves_multitask.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatter(predictions, labels, task_name):
    """
    绘制预测值与真实值的散点图
    
    Args:
        predictions: 模型预测值
        labels: 真实标签值
        task_name: 任务名称 ('cycle' 或 'area')
    """
    plt.figure(figsize=(12, 10)) # 增大图像尺寸
    plt.scatter(labels, predictions, alpha=0.6, s=150) # 增大点的大小
    plt.plot([min(labels.min(), predictions.min()), max(labels.max(), predictions.max())], 
             [min(labels.min(), predictions.min()), max(labels.max(), predictions.max())], 
             'r--', alpha=0.8, linewidth=2) # 增大线条宽度
    
    if task_name == 'cycle':
        title = 'Remaining Cycles: Predicted vs Actual'
        xlabel = 'Actual Remaining Cycles'
        ylabel = 'Predicted Remaining Cycles'
    else:
        title = 'Spalling Area: Predicted vs Actual'
        xlabel = 'Actual Spalling Area'
        ylabel = 'Predicted Spalling Area'
    
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=16) # 增大刻度字体
    plt.yticks(fontsize=16) # 增大刻度字体
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 计算并显示评估指标
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    if np.sum((labels - np.mean(labels)) ** 2) == 0: # 避免除以零
        r2 = 0
    else:
        r2 = 1 - np.sum((predictions - labels) ** 2) / np.sum((labels - np.mean(labels)) ** 2)
    
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), fontsize=18) # 增大文本字体
    
    # 保存图片
    plt.savefig(os.path.join(Config.log_dir, f'{task_name}_predictions_multitask.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()