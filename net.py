import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SpectralDataset(Dataset):
    def __init__(self, X, y):
        # 转换为PyTorch张量并添加通道维度
        self.X = torch.FloatTensor(X).unsqueeze(1)  # shape: (N, 1, n_bands)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpectralCNN_coral(nn.Module):
    def __init__(self, n_classes):  # 移除n_classes参数
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes - 1)  # CORAL需要4个输出节点
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x, self.classifier(x)

class SpectralCNN(nn.Module):
    def __init__(self,  n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),  # 保持维度
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),


            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = x.view(x.size(0), -1)  # 展平
        return x, self.classifier(x)  # 分类


class CompactCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            # 保持序列长度
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # [bs,16,6]
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # [bs,32,6]
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(4)  # 可控压缩 [bs,32,4]
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = x.view(x.size(0), -1)  # 展平
        return x, self.classifier(x)  # 分类
class SpectralCNN_shap(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # 添加通道维度 (batch_size, 1, n_bands)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class SpectralCNN_2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 保持维度
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(2),


            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = x.view(x.size(0), -1)  # 展平
        return x, self.classifier(x)  # 分类


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.shortcut:
            residual = self.shortcut(residual)
        x += residual
        return self.gelu(x)





class CNN_LSTM(nn.Module):
    def __init__(self, n_bands, n_classes, lstm_hidden_size=128, lstm_num_layers=1):  # 调整hidden_size为128以匹配原网络
        super().__init__()
        # ----------------------------------
        # CNN特征提取部分
        # ----------------------------------
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # ----------------------------------
        # LSTM时序建模部分
        # ----------------------------------
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,  # 与原始CNN的embedding维度（128）对齐
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # ----------------------------------
        # 分类器部分
        # ----------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # CNN提取空间特征
        cnn_features = self.cnn(x)  # 输出形状: [batch, 128, n_bands]

        # 调整维度输入LSTM
        lstm_input = cnn_features.permute(0, 2, 1)  # [batch, n_bands, 128]

        # LSTM处理时序
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)  # lstm_out形状: [batch, n_bands, 128]

        # 提取embedding（取最后一个时间步）
        embedding = lstm_out[:, -1, :]  # [batch, 128]

        # 分类结果
        logits = self.classifier(embedding)

        return embedding, logits  # 与原网络输出形式一致




class EuclideanMetricLoss(nn.Module):
    def __init__(self, intra_weight=1.0, inter_weight=0.8, eps=1e-8):
        super().__init__()
        self.intra_weight = intra_weight  # 类内损失权重
        self.inter_weight = inter_weight  # 类间损失权重
        self.eps = eps

        # 用于稳定计算的缓冲张量
        self.register_buffer('zero_tensor', torch.tensor(0.0))

    def forward(self, features, labels):
        """
        改进后的度量损失函数
        Args:
            features: 特征张量 [B, D]
            labels: 标签张量 [B]
        Returns:
            loss: 组合后的损失值
        """
        # 获取唯一类别信息
        unique_labels, counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_labels)

        # 单类别情况处理
        if n_classes < 2:
            return self.zero_tensor

        # 类中心计算
        centers = torch.stack([
            features[labels == lbl].mean(dim=0)
            for lbl in unique_labels
        ])  # [C, D]

        # 类内距离计算 (所有样本到对应中心的距离)
        intra_dists = torch.norm(
            features - centers[torch.searchsorted(unique_labels, labels)],
            dim=1
        )  # [B]
        intra_loss = intra_dists.mean()

        # 类间距离计算 (所有中心之间的最小距离)
        c_diff = centers.unsqueeze(1) - centers.unsqueeze(0)  # [C, C, D]
        inter_dists = torch.norm(c_diff, dim=2)  # [C, C]

        # 排除对角线元素
        triu_mask = ~torch.eye(n_classes, dtype=torch.bool, device=features.device)
        valid_inter_dists = inter_dists[triu_mask]

        # 类间损失：最大化最小间距
        inter_loss = -torch.log(valid_inter_dists.min())

        # 组合损失
        total_loss = (
                self.intra_weight * intra_loss +
                self.inter_weight * inter_loss
        )

        return total_loss


class EuclideanMetricLoss_pro(nn.Module):
    def __init__(self, margin=2.0, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.eps = eps
        self.register_buffer('zero', torch.tensor(0.0))

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)  # L2归一化

        unique_labels, counts = torch.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            return self.zero

        # 类中心计算 (归一化后)
        centers = torch.stack([
            features[labels == lbl].mean(dim=0)
            for lbl in unique_labels
        ])

        # 类内平方距离
        intra_dists = torch.sum(
            (features - centers[torch.searchsorted(unique_labels, labels)]) ** 2,
            dim=1
        )
        intra_loss = intra_dists.mean()

        # 类间距离矩阵
        c_dist = torch.cdist(centers, centers, p=2)  # 更高效的距离计算
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
        valid_dists = c_dist[mask]

        # 带Margin的类间损失
        min_inter = valid_dists.min()
        inter_loss = F.relu(self.margin - min_inter)

        # 自适应权重
        safe_ratio = (min_inter.detach() / self.margin).clamp(0, 1)
        intra_weight = 1.0 + 2.0 * (1 - safe_ratio)  # 类内不足时加强聚合
        inter_weight = 2.0 * safe_ratio  # 类间足够时降低惩罚

        return intra_weight * intra_loss + inter_weight * inter_loss


class KernelSimilarityLoss(nn.Module):
    def __init__(self, margin=0.5, kernel='gaussian', sigma=1.0):
        super().__init__()
        self.margin = margin
        self.kernel_type = kernel
        self.sigma = sigma
        self.register_buffer('zero', torch.tensor(0.0))

        # 初始化核函数
        if kernel == 'gaussian':
            self.kernel = self.gaussian_similarity
        elif kernel == 'linear':
            self.kernel = self.linear_similarity
        else:
            raise ValueError("Unsupported kernel type")

    def gaussian_similarity(self, x, y):
        """修正后的高斯相似度函数（生成二维矩阵）"""
        pairwise_dist = torch.cdist(x, y, p=2)  # 计算两两距离
        return torch.exp(-pairwise_dist ** 2 / (2 * self.sigma ** 2))

    def linear_similarity(self, x, y):
        """修正后的线性相似度函数（保持二维输出）"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        return torch.mm(x, y.t())

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)

        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return self.zero

        # 类内相似度计算
        intra_loss = 0.0
        valid_classes = 0
        for lbl in unique_labels:
            class_feats = features[labels == lbl]
            if len(class_feats) < 2:
                continue  # 跳过单样本类
            # 生成NxN相似度矩阵
            sim_matrix = self.kernel(class_feats, class_feats)
            # 创建掩码排除对角线
            mask = ~torch.eye(len(class_feats), dtype=torch.bool, device=features.device)
            intra_loss += sim_matrix[mask].mean()
            valid_classes += 1

        if valid_classes == 0:
            return self.zero
        intra_loss /= valid_classes

        # 类间相似度计算
        centers = torch.stack([features[labels == lbl].mean(0) for lbl in unique_labels])
        inter_sim = self.kernel(centers, centers)
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
        max_inter = inter_sim[mask].max()
        inter_loss = F.relu(max_inter - self.margin)

        # 自适应权重平衡
        balance_factor = 1.0 + (max_inter.detach() / self.margin).clamp(0, 1)
        return balance_factor * inter_loss - intra_loss

# class EuclideanMetricLoss_pro_2(nn.Module):
#     def __init__(self, intra_margin=0.3, inter_margin=2.0, alpha=0.7):
#         """
#         参数说明：
#         intra_margin : 类内样本最大允许距离阈值
#         inter_margin : 类间中心最小间隔阈值
#         alpha : 类内/类间损失平衡系数
#         """
#         super().__init__()
#         self.intra_margin = intra_margin
#         self.inter_margin = inter_margin
#         self.alpha = alpha
#         self.eps = 1e-6
#
#     def forward(self, features, labels):
#         features = F.normalize(features, p=2, dim=1)  # L2归一化
#
#         # 增强类内约束
#         intra_loss = self._enhanced_intra_loss(features, labels)
#
#         # 增强类间约束
#         inter_loss = self._enhanced_inter_loss(features, labels)
#
#         return self.alpha * intra_loss + (1 - self.alpha) * inter_loss
#
#     def _enhanced_intra_loss(self, features, labels):
#         """改进的类内损失：双重约束策略"""
#         unique_labels = torch.unique(labels)
#         total_loss = 0.0
#
#         for lbl in unique_labels:
#             mask = labels == lbl
#             class_feats = features[mask]
#             if class_feats.size(0) < 2:
#                 continue
#
#             # 约束1：类内样本到中心的平均距离
#             center = class_feats.mean(dim=0)
#             dist_to_center = torch.norm(class_feats - center, dim=1)
#             loss_center = F.relu(dist_to_center - self.intra_margin).mean()
#
#             # 约束2：类内最远样本对距离
#             pairwise_dist = torch.cdist(class_feats, class_feats, p=2)
#             max_dist = pairwise_dist.max()
#             loss_pair = F.relu(max_dist - 2 * self.intra_margin)
#
#             total_loss += loss_center + loss_pair
#
#         return total_loss / len(unique_labels)
#
#     def _enhanced_inter_loss(self, features, labels):
#         """改进的类间损失：全矩阵约束"""
#         centers = []
#         unique_labels = torch.unique(labels)
#         if len(unique_labels) < 2:
#             return 0.0
#
#         # 计算各类中心
#         for lbl in unique_labels:
#             mask = labels == lbl
#             centers.append(features[mask].mean(dim=0))
#         centers = torch.stack(centers)
#
#         # 计算类间距离矩阵
#         inter_dist = torch.cdist(centers, centers, p=2)
#         mask = torch.eye(len(centers), dtype=bool, device=features.device)
#         valid_dist = inter_dist[~mask]
#
#         # 双重约束策略
#         min_dist = valid_dist.min()
#         avg_dist = valid_dist.mean()
#
#         # 约束1：最小类间距超过margin
#         loss_min = F.relu(self.inter_margin - min_dist)
#
#         # 约束2：平均类间距动态调整
#         loss_avg = F.relu((self.inter_margin + 0.5) - avg_dist)
#
#         return loss_min + loss_avg
#
#     def adaptive_weight_scheduler(self, epoch):
#         """动态调整权重系数"""
#         if epoch < 30:  # 初始阶段侧重类内聚合
#             self.alpha = 0.9
#         elif epoch < 60:  # 中期平衡发展
#             self.alpha = 0.7
#         else:  # 后期加强类间分离
#             self.alpha = 0.5

class RobustFocalMetricLoss(nn.Module):
    def __init__(self, margin=5.0, gamma=2.0, alpha=0.25, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.gamma = gamma  # 困难样本聚焦强度
        self.alpha = alpha  # 分类置信度调节因子
        self.eps = eps
        self.register_buffer('zero', torch.tensor(0.0))

    def forward(self, features, labels, logits=None):
        """
        新增logits参数用于获取分类置信度
        """
        features = F.normalize(features, p=2, dim=1)
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return self.zero

        # 类中心计算
        centers = torch.stack([features[labels == lbl].mean(dim=0) for lbl in unique_labels])

        # ========== 动态权重计算 ==========
        intra_dists = torch.sum((features - centers[torch.searchsorted(unique_labels, labels)]) ** 2, dim=1)
        dist_normalized = intra_dists / (intra_dists.max() + self.eps)

        # 关键改进1：分类置信度感知权重
        if logits is not None:
            probs = torch.softmax(logits.detach(), dim=1)
            conf = probs[torch.arange(len(labels)), labels]  # 正确类别的置信度
            conf_weight = (1 - conf.clamp(0.1, 0.9)) ** 0.5  # 置信度越低 → 权重越高
        else:
            conf_weight = 1.0

        # 关键改进2：双阶段权重函数
        focal_weights = torch.where(
            dist_normalized > 0.7,  # 阈值可调
            (dist_normalized ** self.gamma) * conf_weight,  # 离群样本：降低权重
            (1 - dist_normalized) ** self.gamma  # 边界样本：正常处理
        )

        intra_loss = (focal_weights.detach() * intra_dists).mean()

        # ========== 类间损失改进 ==========
        c_dist = torch.cdist(centers, centers, p=2)
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
        valid_dists = c_dist[mask]

        # 关键改进3：安全间隔动态调整
        min_inter = valid_dists.min()
        safe_margin = self.margin * (1 + 0.5 * torch.sigmoid(-intra_loss.detach()))  # 类内损失大时增大margin
        inter_loss = F.relu(safe_margin - min_inter)

        # ========== 自适应平衡 ==========
        safe_ratio = (min_inter.detach() / safe_margin).clamp(0, 1)
        intra_weight = 1.0 + 2.0 * (1 - safe_ratio)
        inter_weight = 2.0 * safe_ratio

        return intra_weight * intra_loss + inter_weight * inter_loss

class FocalEuclideanMetricLoss(nn.Module):
    def __init__(self, margin=2.0, gamma=2.0, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.gamma = gamma  # 困难样本聚焦参数
        self.eps = eps
        self.register_buffer('zero', torch.tensor(0.0))

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)

        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return self.zero

        # 类中心计算
        centers = torch.stack([
            features[labels == lbl].mean(dim=0)
            for lbl in unique_labels
        ])

        # ========== 类内损失（Focal加权） ==========
        # 计算每个样本到类中心的距离
        intra_dists = torch.sum(
            (features - centers[torch.searchsorted(unique_labels, labels)]) ** 2,
            dim=1
        )

        # 动态计算困难样本权重
        dist_normalized = intra_dists / (intra_dists.max() + self.eps)  # 归一化距离 [0,1]
        # focal_weights = (1 - dist_normalized.detach()) ** self.gamma  # 距离越大→权重越高
        focal_weights = (dist_normalized.detach()) ** self.gamma

        intra_loss = (focal_weights * intra_dists).mean()

        # ========== 类间损失（困难样本感知） ==========
        c_dist = torch.cdist(centers, centers, p=2)
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
        valid_dists = c_dist[mask]

        # 寻找最难样本对应的类间距离
        min_inter = valid_dists.min()
        inter_loss = F.relu(self.margin - min_inter)

        # ========== 自适应平衡系数 ==========
        safe_ratio = (min_inter.detach() / self.margin).clamp(0, 1)
        intra_weight = 1.0 + 2.0 * (1 - safe_ratio)
        inter_weight = 2.0 * safe_ratio

        return intra_weight * intra_loss + inter_weight * inter_loss

class EuclideanMetricLoss_pro_kernell(nn.Module):
    def __init__(self, margin=2.0, sigma=1.0, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.eps = eps
        self.register_buffer('zero', torch.tensor(0.0))

        # 可学习的核参数（RBF带宽）
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))  # 对数变换保证正数

    def rbf_kernel(self, x, y):
        """
        计算RBF核矩阵 (兼容高维张量)
        Args:
            x: (N, D)
            y: (M, D)
        Returns:
            kernel_matrix: (N, M)
        """
        sigma = torch.exp(self.log_sigma)  # 确保sigma>0
        pairwise_dist = torch.cdist(x, y, p=2)  # 欧氏距离
        return torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2 + self.eps))

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)  # L2归一化

        unique_labels, counts = torch.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            return self.zero

        # 类中心计算（核空间隐式映射）
        centers = torch.stack([
            features[labels == lbl].mean(dim=0)
            for lbl in unique_labels
        ])

        # ========== 类内损失（核空间距离） ==========
        # 计算每个样本与类中心的核相似度
        intra_sim = self.rbf_kernel(features, centers[torch.searchsorted(unique_labels, labels)])
        intra_loss = 1.0 - intra_sim.diag().mean()  # 相似度转距离：距离 = 1 - 相似度

        # ========== 类间损失（核空间间隔） ==========
        # 计算类间核相似度矩阵
        inter_sim = self.rbf_kernel(centers, centers)
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
        valid_sims = inter_sim[mask]

        # 最大相似度对应最小距离
        max_inter_sim = valid_sims.max()  # 相似度越大，距离越小
        inter_loss = F.relu(max_inter_sim - (1.0 - self.margin))  # 相似度超过阈值时惩罚

        # ========== 自适应权重 ==========
        safe_ratio = (1.0 - max_inter_sim.detach()).clamp(0, 1) / self.margin
        intra_weight = 1.0 + 2.0 * (1 - safe_ratio)
        inter_weight = 2.0 * safe_ratio

        return intra_weight * intra_loss + inter_weight * inter_loss
#
# class EuclideanMetricLoss_pro_2(nn.Module):
#     def __init__(self, margin=2.0, eps=1e-6):
#         super().__init__()
#         self.margin = margin
#         self.eps = eps
#         self.register_buffer('zero', torch.tensor(0.0))
#
#     def forward(self, features, labels):
#         features = F.normalize(features, p=2, dim=1)  # L2归一化
#
#         unique_labels, counts = torch.unique(labels, return_counts=True)
#         if len(unique_labels) < 2:
#             return self.zero
#
#         # 类中心计算 (归一化后)
#         centers = torch.stack([
#             features[labels == lbl].mean(dim=0)
#             for lbl in unique_labels
#         ])
#
#         # 类内平方距离
#         intra_dists = torch.sum(
#             (features - centers[torch.searchsorted(unique_labels, labels)]) ** 2,
#             dim=1
#         )
#         intra_loss = intra_dists.mean()
#
#         # 类间距离矩阵
#         c_dist = torch.cdist(centers, centers, p=2)  # 更高效的距离计算
#         mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
#         valid_dists = c_dist[mask]
#
#         # 带Margin的类间损失
#         min_inter = valid_dists.min()
#
#         k = 3  # 考虑前3个最近类间距离
#         topk_values = torch.topk(valid_dists, k=k, largest=False).values
#         inter_loss = F.relu(self.margin - topk_values).mean()
#
#
#         # 自适应权重
#         safe_ratio = (min_inter.detach() / self.margin).clamp(0, 1)
#         intra_weight = 1.0 + 2.0 * (1 - safe_ratio)  # 类内不足时加强聚合
#         inter_weight = 2.0 * safe_ratio  # 类间足够时降低惩罚
#
#         return intra_weight * intra_loss + inter_weight * inter_loss

class CosineMetricLoss(nn.Module):
    def __init__(self, margin=0.4, eps=1e-6):
        super().__init__()
        self.margin = margin  # 余弦空间中的间隔（范围：-1到1）
        self.eps = eps
        self.register_buffer('zero', torch.tensor(0.0))

    def forward(self, features, labels):
        # 特征归一化（关键步骤）
        features = F.normalize(features, p=2, dim=1)  # L2归一化

        unique_labels, counts = torch.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            return self.zero

        # 计算类中心（已归一化）
        centers = torch.stack([
            features[labels == lbl].mean(dim=0)
            for lbl in unique_labels
        ])
        centers = F.normalize(centers, p=2, dim=1)  # 中心再次归一化

        # ------------------- 类内损失 -------------------
        # 计算样本与所属类中心的余弦相似度
        intra_sim = torch.sum(
            features * centers[torch.searchsorted(unique_labels, labels)],
            dim=1
        )
        # 转换为损失（最大化相似度）
        intra_loss = torch.mean(1.0 - intra_sim)  # 1 - cosθ ∈ [0, 2]

        # ------------------- 类间损失 -------------------
        # 计算类间余弦相似度矩阵
        c_sim = torch.mm(centers, centers.T)  # [C, C]

        # 排除对角线并找到最大相似度
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
        max_inter_sim = torch.max(c_sim[mask])  # 找最接近的类对

        # 间隔惩罚（当类间相似度高于margin时惩罚）
        inter_loss = F.relu(max_inter_sim - self.margin)

        # ------------------- 动态权重 -------------------
        # 相似度安全系数计算
        safe_ratio = ((max_inter_sim.detach() - self.margin) /
                      (1.0 - self.margin)).clamp(0, 1)

        # 类内权重：当类间太近时加强聚合
        intra_weight = 1.0 + 2.0 * safe_ratio
        # 类间权重：当类间已分离时降低惩罚
        inter_weight = 2.0 * (1.0 - safe_ratio)

        return intra_weight * intra_loss + inter_weight * inter_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        # 特别针对健康(0)和无症状(1)类
        mask = torch.isin(labels, torch.tensor([0, 1], device=labels.device))
        if mask.sum() < 2:
            return torch.tensor(0.0, device=features.device)

        # 提取目标类别特征
        target_features = features[mask]
        target_labels = labels[mask]

        # 计算类内/类间距离
        dist_matrix = torch.cdist(target_features, target_features, p=2)
        same_class = target_labels.unsqueeze(0) == target_labels.unsqueeze(1)
        diff_class = ~same_class

        intra_loss = dist_matrix[same_class].mean()
        inter_loss = F.relu(self.margin - dist_matrix[diff_class].min())

        return intra_loss + inter_loss



def extract_data_from_dataloader(dataloader):
    """从DataLoader中提取numpy数据"""
    features = []
    labels = []
    for batch in dataloader:
        x, y = batch
        features.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


class SpectralDataset_LSTM(Dataset):
    def __init__(self, X, y):
        # 调整为LSTM需要的形状 (样本数, 序列长度, 特征维度)
        self.X = torch.FloatTensor(X).unsqueeze(-1)  # shape: (N, n_bands, 1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SpectralLSTM_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpectralLSTM_0, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 全连接层（双向需要2*hidden_size）
        self.fc = nn.Linear(2 * hidden_size, num_classes)

        # 特征维度（用于对比学习）
        self.feature_dim = 2 * hidden_size

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)

        # 取双向最后时刻的隐藏状态并拼接
        forward_feature = hn[-2, :, :]
        backward_feature = hn[-1, :, :]
        features = torch.cat((forward_feature, backward_feature), dim=1)

        # 分类输出
        outputs = self.fc(features)
        return features, outputs

class SpectralLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpectralLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 全连接层（双向需要2*hidden_size）
        self.fc = nn.Linear(2 * hidden_size, num_classes)

        # 特征维度（用于对比学习）
        self.feature_dim = 2 * hidden_size

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)

        # 取双向最后时刻的隐藏状态并拼接
        forward_feature = hn[-2, :, :]
        backward_feature = hn[-1, :, :]
        features = torch.cat((forward_feature, backward_feature), dim=1)

        # 分类输出
        outputs = self.fc(features)
        return features, outputs


class MultiBranchBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 各分支保持输入输出通道一致
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 15, padding=7),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 9, padding=4),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, padding=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        # 注意力机制通道匹配
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(3 * in_channels, 3 * in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        concat = torch.cat([b1, b2, b3], dim=1)  # [B, 3*in_channels, L]
        att = self.attention(concat)  # [B, 3*in_channels, 1]
        return concat * att.expand_as(concat)  # [B, 3*in_channels, L]


class CenterMetricLoss(nn.Module):
    def __init__(self, num_classes, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature  # 温度系数用于缩放logits

    def forward(self, features, labels):
        """
        参数：
        features : 形状 [bs, ch] 的特征张量
        labels   : 形状 [bs] 的标签张量
        返回：
        loss    : 标量损失值
        logits  : 用于分类的logits
        """
        # 计算类中心
        centers = self.compute_centers(features, labels)  # [num_classes, ch]

        # 计算样本到各中心的距离（使用欧氏距离平方）
        distances = self.pairwise_distance(features, centers)  # [bs, num_classes]

        # 转换为相似度logits（距离越小相似度越高）
        logits = -distances / self.temperature  # [bs, num_classes]

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_centers(self, features, labels):
        # 创建one-hot编码矩阵 [bs, num_classes]
        one_hot = F.one_hot(labels, self.num_classes).float()  # [bs, C]

        # 统计每个类的样本数 [num_classes]
        counts = one_hot.sum(dim=0)  # [C]

        # 计算类中心 [num_classes, feat_dim]
        centers = torch.matmul(one_hot.T, features)  # [C, ch]
        centers = centers / (counts.unsqueeze(1) + 1e-8)  # 防止除零

        return centers

    def pairwise_distance(self, x, centers):
        """
        计算每个样本到所有类中心的欧氏距离平方
        x      : [bs, ch]
        centers: [C, ch]
        返回：
        distances: [bs, C]
        """
        # 展开维度用于广播计算
        x = x.unsqueeze(1)  # [bs, 1, ch]
        centers = centers.unsqueeze(0)  # [1, C, ch]

        # 计算平方差并求和
        squared_diff = (x - centers).pow(2)  # [bs, C, ch]
        distances = squared_diff.sum(dim=2)  # [bs, C]

        return distances


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.branch3 = nn.Conv1d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)


class ImprovedSpectralCNN(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            MultiScaleConv(32),  # 多尺度特征融合
            nn.BatchNorm1d(96),
            nn.ReLU(),

            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            ChannelAttention(128),  # 通道注意力
            SpatialAttention()  # 空间注意力
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=-1)  # 全局平均池化
        return x, self.classifier(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_cent=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_cent = lambda_cent

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        features = F.normalize(features, p=2, dim=1)  # L2归一化

        # 计算中心损失
        centers_batch = self.centers[labels]
        loss_cent = torch.sum((features - centers_batch) ** 2) / batch_size

        # 更新中心（仅训练时）
        if self.training:
            delta = torch.zeros_like(self.centers)
            for i in range(batch_size):
                delta[labels[i]] += (centers_batch[i] - features[i]) / (1 + torch.sum(labels == labels[i]))
            self.centers.data -= self.lambda_cent * delta

        return loss_cent


class EnhancedEuclideanMetricLoss(nn.Module):
    def __init__(self, margin=5.0, alpha=0.5):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # 类内-类间损失的平衡系数
        self.eps = 1e-9

    def forward(self, features, labels):
        """
        参数：
            features : 模型输出的特征向量 (batch_size, feat_dim)
            labels : 样本标签 (batch_size,)
        """
        # 计算类内紧凑性损失
        intra_loss = self._intra_class_loss(features, labels)

        # 计算类间分离性损失
        inter_loss = self._inter_class_loss(features, labels)

        # 组合损失
        total_loss = intra_loss + self.alpha * inter_loss
        return total_loss

    def _intra_class_loss(self, features, labels):
        """类内紧凑性损失"""
        unique_labels = torch.unique(labels)
        loss = 0.0

        for lbl in unique_labels:
            # 计算类内样本间的平均距离
            mask = (labels == lbl)
            class_features = features[mask]
            if class_features.size(0) < 2:
                continue  # 跳过单样本类

            # 计算所有样本对的距离
            dist = torch.cdist(class_features, class_features, p=2)
            n_pairs = dist.size(0) * (dist.size(1) - 1)  # 排除对角线
            loss += torch.sum(dist) / (n_pairs + self.eps)

        return loss / len(unique_labels)

    def _inter_class_loss(self, features, labels):
        """类间分离性损失"""
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return 0.0  # 单类别无类间损失

        # 计算类中心
        centers = []
        for lbl in unique_labels:
            mask = (labels == lbl)
            centers.append(features[mask].mean(dim=0))
        centers = torch.stack(centers)  # (n_classes, feat_dim)

        # 计算类间距离
        inter_dist = torch.cdist(centers, centers, p=2)
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=features.device)
        valid_dist = inter_dist[mask]

        # 使用margin-based损失
        loss = torch.mean(torch.relu(self.margin - valid_dist))
        return loss


class DynamicCompositeLoss(nn.Module):
    def __init__(self, ce_weight=1.0, metric_weight=0.3, temp=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.metric = EnhancedEuclideanMetricLoss()
        self.ce_weight = ce_weight
        self.metric_weight = metric_weight
        self.temp = temp  # 温度系数

    def forward(self, features, outputs, labels):
        # 基础损失计算
        ce_loss = self.ce(outputs, labels)
        metric_loss = self.metric(features, labels)

        # 动态调整权重（基于特征相似度）
        with torch.no_grad():
            sim_matrix = torch.mm(features, features.T)
            avg_sim = torch.mean(sim_matrix)
            dynamic_weight = torch.sigmoid((avg_sim - self.temp) / 0.1)

        total_loss = (self.ce_weight * ce_loss +
                      self.metric_weight * dynamic_weight * metric_loss)
        return total_loss


class KernelMetricLoss(nn.Module):
    def __init__(self, kernel_type='rbf', gamma=1.0, margin=5.0, alpha=0.5):
        """
        参数：
            kernel_type : 核函数类型 ('rbf', 'poly', 'linear')
            gamma : RBF核的带宽参数
            margin : 类间距离的margin阈值
            alpha : 类内/类间损失平衡系数
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.margin = margin
        self.alpha = alpha
        self.eps = 1e-9

    def _kernel_distance(self, X, Y):
        """计算核空间中的距离矩阵"""
        if self.kernel_type == 'rbf':
            # RBF核距离公式：sqrt(2(1 - K(x,y)))
            pairwise_dist = torch.cdist(X, Y, p=2)
            K = torch.exp(-self.gamma * pairwise_dist.pow(2))
            return torch.sqrt(2 * (1 - K + self.eps))
        elif self.kernel_type == 'poly':
            # 多项式核：K(x,y) = (gamma<x,y> + c)^d
            # 这里使用简化的二次核距离
            K = torch.matmul(X, Y.T)
            return torch.sqrt(K.diag().unsqueeze(1) + K.diag().unsqueeze(0) - 2 * K + self.eps)
        elif self.kernel_type == 'linear':
            # 线性核的等效欧式距离
            return torch.cdist(X, Y, p=2)
        else:
            raise ValueError("Unsupported kernel type")

    def forward(self, features, labels):
        intra_loss = self._intra_class_loss(features, labels)
        inter_loss = self._inter_class_loss(features, labels)
        return intra_loss + self.alpha * inter_loss

    def _intra_class_loss(self, features, labels):
        unique_labels = torch.unique(labels)
        loss = 0.0

        for lbl in unique_labels:
            mask = (labels == lbl)
            class_feats = features[mask]
            if class_feats.size(0) < 2:
                continue

            # 计算核距离矩阵
            dist_matrix = self._kernel_distance(class_feats, class_feats)

            # 排除对角线元素
            mask = torch.eye(dist_matrix.size(0), dtype=torch.bool, device=features.device)
            valid_dist = dist_matrix[~mask]

            loss += valid_dist.mean()

        return loss / len(unique_labels)

    def _inter_class_loss(self, features, labels):
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        # 计算类中心
        centers = []
        for lbl in unique_labels:
            mask = (labels == lbl)
            centers.append(features[mask].mean(dim=0))
        centers = torch.stack(centers)

        # 计算类中心间的核距离
        center_dist = self._kernel_distance(centers, centers)

        # 排除自对角线
        mask = torch.eye(center_dist.size(0), dtype=torch.bool, device=features.device)
        valid_dist = center_dist[~mask]

        # Margin-based损失
        loss = torch.relu(self.margin - valid_dist).mean()
        return loss