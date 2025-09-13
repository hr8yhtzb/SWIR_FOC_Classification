import numpy as np
import time
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import utils
import math
import torch.nn.functional as F
import net
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from net import SpectralDataset, SpectralCNN
from captum.attr import LayerGradCam
from net import  CenterMetricLoss, EuclideanMetricLoss_pro, KernelSimilarityLoss
import random
seed = 42  # 可任意指定，推荐42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保卷积结果一致
    torch.backends.cudnn.benchmark = False     # 关闭自动优化


class AdaptiveTripletLoss(nn.Module):
    def __init__(self, margin=0.3, top_n=3, use_soft_margin=True, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.top_n = top_n
        self.use_soft_margin = use_soft_margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, embeddings, targets):
        device = embeddings.device
        n, d = embeddings.size()
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Distance matrix
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # (n, n)

        # Positive mask (excluding self)
        target_eq = targets.unsqueeze(0) == targets.unsqueeze(1)
        mask_pos = target_eq.fill_diagonal_(False)

        # Top-N hard positives
        FILL = -1e6
        pos_dists = dist_matrix.masked_fill(~mask_pos, FILL)
        top_pos_values, top_pos_indices = torch.topk(pos_dists, k=min(self.top_n, n - 1), dim=1)
        pos_samples = embeddings[top_pos_indices]  # (n, top_n, d)
        pos_centers = pos_samples.mean(dim=1)      # (n, d)

        # Compute class centers (one-hot based)
        classes = torch.unique(targets)
        num_classes = len(classes)
        class_map = {cls.item(): i for i, cls in enumerate(classes)}
        class_ids = torch.tensor([class_map[t.item()] for t in targets], device=device)

        one_hot = F.one_hot(class_ids, num_classes).float()  # (n, C)
        class_counts = one_hot.sum(0).clamp(min=1e-6).unsqueeze(1)  # (C, 1)
        class_sums = one_hot.T @ embeddings                    # (C, d)
        class_centers = class_sums / class_counts              # (C, d)

        # Get anchor's own class center
        own_class_center = class_centers[class_ids]            # (n, d)

        # Compute all centers excluding anchor's own class
        mask_class = torch.ones_like(class_centers, dtype=torch.bool)  # (C, d)
        mask_class[targets] = False  # Mark each anchor's own class to be excluded

        # Broadcast anchors vs all class centers
        d_all = torch.cdist(embeddings, class_centers, p=2)    # (n, C)
        d_all = d_all.scatter(1, class_ids.unsqueeze(1), float('inf'))  # set self-class dist to inf

        d_neg_min, _ = d_all.min(dim=1)  # (n,)

        d_pos = torch.norm(embeddings - pos_centers, p=2, dim=1)

        # Valid mask
        valid_mask = (top_pos_values != FILL).any(dim=1)
        d_pos = d_pos[valid_mask]
        d_neg = d_neg_min[valid_mask]

        # Compute loss
        if self.use_soft_margin:
            loss = torch.log(1 + torch.exp(self.pos_weight * d_pos - self.neg_weight * d_neg))
        else:
            loss = F.relu(self.pos_weight * d_pos - self.neg_weight * d_neg + self.margin)

        return loss.mean()


#============================= 波段选择部分 ================================

def extract_data_from_dataloader(dataloader):
    """从DataLoader中提取numpy数据"""
    features = []
    labels = []
    for batch in dataloader:
        x, y = batch
        features.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels)



# 5_class
data_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\5_CLASS\3.cleaned_data" # processed_data/1.MSC
class_names = ["H", "A", "MI", "MO", "SE"]

AUG_PARAMS = {
    'noise_std': 0.008,
    'random_state': 42
}


cfg_custom_order = {
        'steps': ['1D']
    }

# 五类别 1000-1700nm
train_dataloader, test_dataloader = utils.load_hyper(data_dir, class_names, 0.6, 128,
                                                     dynamic_aug = False, static_aug = True,
                                                     augment_times=5,  # 每个样本生成3个增强版本
                                                     aug_params=AUG_PARAMS,
                                                     preprocess_params=cfg_custom_order)


# 从DataLoader提取数据
X_train, y_train = extract_data_from_dataloader(train_dataloader)
X_test, y_test = extract_data_from_dataloader(test_dataloader)


wavelengths = 1050 + np.arange(X_train.shape[1])  # 生成完整波长数组


# selected_wavelengths_1 = [1604, 1272, 1187, 1416, 1359, 1404] # 6
# selected_wavelengths_1 = [1604, 1272, 1187, 1416, 1359, 1404, 1168] # 7
# selected_wavelengths_1 = [1604, 1272, 1187, 1416, 1359, 1404, 1168, 1489] # 8
# selected_wavelengths_1 = [1604, 1272, 1187, 1416, 1359, 1404, 1168, 1489, 1366] # 9
selected_wavelengths_1 = [1604, 1272, 1187, 1416, 1359, 1404, 1168, 1489, 1366, 1151] # 10



selected_wavelengths = sorted(selected_wavelengths_1)

# 将波长转换为索引
selected_bands = [np.where(wavelengths == wl)[0][0] for wl in selected_wavelengths]

# 验证索引转换结果
print(f"波长对应索引检查：")
for wl, idx in zip(selected_wavelengths, selected_bands):
    print(f"{wl}nm -> 索引{idx} (验证波长：{wavelengths[idx]}nm)")

print("selected_bands",selected_bands)
# 提取特征子集
X_train_sel = X_train[:, selected_bands]
X_test_sel = X_test[:, selected_bands]

# 可视化选择结果
plt.figure(figsize=(12, 5))
plt.plot(wavelengths, X_train.mean(axis=0), alpha=0.6, label='平均光谱')
plt.scatter(wavelengths[selected_bands],
            X_train.mean(axis=0)[selected_bands],
            c='red', s=50, edgecolor='k', zorder=3, label='手动选择波段')

# 添加波长标注
for wl in selected_wavelengths:
    plt.axvline(x=wl, color='orange', linestyle='--', alpha=0.7)
    plt.text(wl+2, 0.1, str(wl), rotation=90, va='bottom', fontsize=9)

plt.title(f'手动选择波段可视化 (共{len(selected_bands)}个波段)')
plt.xlabel('波长(nm)')
plt.ylabel('吸光度')
plt.legend()
plt.tight_layout()
plt.show()

config = {
    'batch_size': 128,
    'lr': 1e-4,
    'n_epochs': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

train_dataset = net.SpectralDataset(X_train_sel, y_train)
test_dataset = net.SpectralDataset(X_test_sel, y_test)

# 初始化模型
n_bands = len(selected_bands)
n_classes = len(class_names)

# model = net.SpectralCNN_2(n_classes).to(config['device'])
model = net.SpectralCNN(n_classes).to(config['device'])

# 初始化损失函数
# 初始化损失函数
criterion = nn.CrossEntropyLoss()
# 初始化方式
criterion_metric = AdaptiveTripletLoss(
        margin=0.3,
        top_n=4,
        use_soft_margin=True,
        pos_weight=1.0,
        neg_weight=1.1
).to(config['device'])


optimizer = optim.Adam(model.parameters(), lr=config['lr'])

# 训练循环
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
# =============================================
# 训练过程
# =============================================
# =============================================
# 修改后的训练循环（含验证阶段）
# =============================================
train_losses = []
val_losses = []       # 新增验证损失记录
val_accuracies = []   # 验证准确率记录
best_val_acc = 0.0
best_model_weights = None

train_start = time.time()
max_epoch = config['n_epochs']

print("开始训练...")
for epoch in range(config['n_epochs']):
    # 训练阶段
    model.train()
    running_loss = 0.0
    # criterion_metric.adaptive_weight_scheduler(epoch)

    for inputs, labels in train_loader:
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        # 前向传播
        features, outputs = model(inputs)
        # 计算复合损失
        loss_ce = criterion(outputs, labels)
        loss_triplet  = criterion_metric(features, labels)
        # current_weight = min(0.3, 0.1 * (epoch // 20))  # 每20个epoch增加0.1
        # loss = loss_ce + current_weight * loss_triplet
        # loss = loss_ce + 0.1 * loss_metric

        # 动态权重调整策略
        if epoch < 50:
            loss = loss_ce  # 初始阶段仅用交叉熵
        elif 50 <= epoch < 100:
            loss = loss_ce + 0.1 * loss_triplet  # 逐步引入
        else:
            loss = loss_ce + 0.2 * loss_triplet  # 稳定阶段


        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)


    # 每5个epoch打印进度
    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{max_epoch}] "
              f"Train Loss: {epoch_loss:.4f} | "
              f"LR: {current_lr:.2e} | ")

# 训练结束后加载最佳模型
train_time = time.time() - train_start


# =============================================
plt.figure(figsize=(8, 5))  # 黄金分割比例
ax = plt.axes()

# ========== 可视化部分修改 ==========
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='#2c7bb6', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='#d7191c', linewidth=2, linestyle='--')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='#2ca25f', linewidth=2)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
# plt.savefig('training_metrics.png', dpi=300)
plt.show()

# =============================================
# 模型评估
# =============================================
model.eval()
all_preds = []
all_labels = []

test_start = time.time()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(config['device'])
        _, outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
test_time = time.time() - test_start
# 转换为numpy数组
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =============================================
#---------------------保存文件-------------------------
# =============================================


# from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3, 4])
# 输出健康(0)与无症状(1)的混淆情况
print("健康→健康:", cm[0,0], " 健康→无症状:", cm[0,1])
print("无症状→无症状:", cm[1,1], " 无症状→健康:", cm[1,0])
print(cm)
# =============================================
# 结果可视化
# =============================================
# 1. 文本报告
print("\n" + "=" * 50)
print(f"CNN分类结果（使用{len(selected_bands)}个波段）")
print("=" * 50)
print(f"准确率: {accuracy_score(all_labels, all_preds):.4f}")
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


print("\n" + "="*50)

print(f"\n模型训练总耗时: {train_time:.2f} 秒")
print(f"模型推理耗时: {test_time:.4f} 秒")
print(f"总建模时间 (训练+推理): {train_time + test_time:.2f} 秒")
print("="*50)

# 可视化参数
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# 创建混淆矩阵 DataFrame（文字输出用）
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\n混淆矩阵（文字形式）：")
print(cm_df)

# 保存文字混淆矩阵为 Excel
cm_df.to_excel('confusion_matrix_text.xlsx', index=True)

# 画图并保存高质量混淆矩阵图片
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
            linewidths=0.5, linecolor='gray', annot_kws={"size": 14})
plt.title(f'Confusion Matrix (with {len(selected_bands)} Bands)', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_CNN.png', dpi=600)  # 高分辨率适合出版
plt.show()
