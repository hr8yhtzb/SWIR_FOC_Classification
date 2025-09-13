from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import utils
import seaborn as sns
import pandas as pd
import time
import net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from net import  EuclideanMetricLoss
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



SPA_CACHE_FILE = "5_CLASS_spa_6_selected_bands.npy"
class SPA:
    def __init__(self, max_vars=15):
        self.max_vars = max_vars
        self.selected_vars = []

    def _projection_operation(self, X_sub, k_local):
        """ 修正后的投影运算（使用子矩阵局部索引） """
        X_k = X_sub[:, [k_local]]  # 使用局部索引
        X_rest = np.delete(X_sub, k_local, axis=1)

        if X_rest.shape[1] == 0:
            return X_k
        P = X_rest @ np.linalg.pinv(X_rest) @ X_k
        return X_k - P

    def fit(self, X, y, start_wavelength = 1050):
        X = StandardScaler().fit_transform(X)
        X = X.reshape(X.shape[0], -1)
        n_samples, n_vars = X.shape

        self.wavelengths = start_wavelength + np.arange(n_vars)
        # 初始化选择
        pls = PLSRegression(n_components=1)
        pls.fit(X, y)
        start_var = np.argmax(np.abs(pls.x_weights_[:, 0]))
        self.selected_vars = [start_var]
        X_candidate = np.delete(np.arange(n_vars), start_var)

        # 迭代修正
        for _ in range(1, min(self.max_vars, n_vars - 1)):
            proj_values = []
            for k in X_candidate:
                # 构造子矩阵并获取局部索引
                sub_matrix = X[:, self.selected_vars + [k]]
                k_local = len(self.selected_vars)  # 子矩阵最后一列的索引
                X_proj = self._projection_operation(sub_matrix, k_local)
                proj_values.append(np.linalg.norm(X_proj))

            # 选择最优k
            best_idx = np.argmax(proj_values)
            selected_k = X_candidate[best_idx]
            self.selected_vars.append(selected_k)
            X_candidate = np.setdiff1d(X_candidate, selected_k)

        return self


def run_or_load_spa(X_train, y_train, cache_file=SPA_CACHE_FILE):
    """运行SPA或加载缓存结果"""
    if os.path.exists(cache_file):
        print(f"检测到缓存文件 {cache_file}，直接加载结果...")
        data = np.load(cache_file, allow_pickle=True).item()
        selected_bands = data['selected_bands']
        print(f"加载的波段索引：{selected_bands}")
        print([x + 1050 for x in selected_bands])
        return selected_bands
    else:
        print("未找到缓存文件，执行SPA算法...")
        spa = SPA(max_vars=6)

        # X_train = StandardScaler().fit_transform(X_train)  # 数据标准化
        spa.fit(X_train, y_train)
        selected_bands = sorted(spa.selected_vars)
        np.save(cache_file, {
            'selected_bands': selected_bands
        })
        print(f"SPA结果已保存到 {cache_file}")
        return selected_bands

# 1000-1700nm
# data_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2/YuanYi/pro_data/Cleaned_data"  # processed_data/1.MSC
# class_names = ["Healthy", "Asymptomatic", "Moderate_infected", "Severely_infected"]


# # 无症状细分
# data_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2/YuanYi/2.Asynptomatic/2.cleaned_data"  # processed_data/1.MSC
# class_names = ["AE", "AM", "AL"]

# # 7_class
# data_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\7_CLASS\3.cleaned_data" # processed_data/1.MSC
# class_names = ["H", "AE", "AM", "AL", "MI", "MO", "SE"]

# 5_class
data_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\5_CLASS\3.cleaned_data" # processed_data/1.MSC
class_names = ["H", "A", "MI", "MO", "SE"]


AUG_PARAMS = {
    'noise_std': 0.008,
    'random_state': 42
}


cfg_1D_only = {
        'steps': ['1D']
    }

cfg_Asynptomatic_order = {
        'steps': ['MSC','1D'],
        'MSC': True
        # 'SG': {'window': 21, 'poly_order': 3}
    }

cfg_custom_order = {
        'steps': ['MSC', 'DeTrend', '1D'],
        'MSC': True,
        'DeTrend': {'poly_order': 1}
    }

train_dataloader, test_dataloader = utils.load_hyper(data_dir, class_names, 0.6, 64, dynamic_aug = False, static_aug = True, augment_times=5,  # 每个样本生成3个增强版本
        aug_params=AUG_PARAMS, preprocess_params=cfg_1D_only)

def extract_data_from_dataloader(dataloader):
    """从DataLoader中提取numpy数据"""
    features = []
    labels = []
    for batch in dataloader:
        x, y = batch
        features.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)



# 从DataLoader提取数据
X_train, y_train = extract_data_from_dataloader(train_dataloader)


start_time = time.time()
selected_bands = run_or_load_spa(X_train, y_train)
spa_time = time.time() - start_time

# print("选择的波段")
# print(list(selected_bands))
# print(list(1050 + selected_bands))

wavelengths = np.linspace(1050, 1650, X_train.shape[1])  # 示例波长范围
selected_wavelengths = wavelengths[selected_bands]

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 计算健康样本平均光谱
healthy_idx = np.where(y_train == 0)[0]
mean_spectrum = X_train[healthy_idx].mean(axis=0)

plt.figure(figsize=(12, 6), dpi=300)

# 绘制光谱曲线
plt.plot(wavelengths, mean_spectrum,
         color='#2c7bb6', linewidth=2, label='SG+1D Processed')

# 标注SPA选择波段
for band in selected_bands:
    plt.axvline(x=wavelengths[band],
                color='#d7191c', linestyle='--',
                linewidth=1.0, alpha=0.7)

# # 突出前3个重要波段
# top3_bands = selected_bands[:3]
# for i, band in enumerate(top3_bands):
#     plt.scatter(wavelengths[band], mean_spectrum[band],
#                 color='#fdae61', edgecolor='k',
#                 s=120, zorder=5, marker=f'${i+1}$')

# 图例和标签
plt.xlabel('Wavelength (nm)', fontsize=12, labelpad=10)
plt.ylabel('1st Derivative (a.u.)', fontsize=12)
plt.title('SPA Selected Bands on Healthy Leaf Spectra', fontsize=14, pad=15)

# 坐标轴优化
plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.grid(True, which='minor', linestyle=':', alpha=0.5)

# 添加颜色条说明
plt.legend(['Processed Spectrum', 'SPA Selected Bands'],
           loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('SPA_selection.png', bbox_inches='tight')
plt.show()



X_test, y_test = extract_data_from_dataloader(test_dataloader)

# 应用SPA波段选择
# 提取SPA波段数据



config = {
    'batch_size': 64,
    'lr': 1e-4,
    'n_epochs': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# =============================================
# 数据准备（接续您的SPA代码）
# =============================================
# 提取测试数据
X_test, y_test = extract_data_from_dataloader(test_dataloader)

# 应用SPA波段选择
X_train_spa = X_train[:, selected_bands]
X_test_spa = X_test[:, selected_bands]

# 标准化处理（注意：光谱数据建议在Dataset内部做归一化）
train_dataset = net.SpectralDataset(X_train_spa, y_train)
test_dataset = net.SpectralDataset(X_test_spa, y_test)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# =============================================
# 模型初始化
# =============================================
n_bands = len(selected_bands)
n_classes = len(class_names)
model = net.SpectralCNN(n_classes).to(config['device'])
criterion = nn.CrossEntropyLoss()
# criterion_metric = EuclideanMetricLoss(temperature=0.5).cuda()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

# =============================================
# 训练过程
# =============================================
train_losses = []
train_start = time.time()
print("开始训练...")
for epoch in range(config['n_epochs']):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        # 前向传播
        features, outputs = model(inputs)
        loss_ce = criterion(outputs, labels)

        loss = loss_ce

        # if epoch <= 100:
        #     loss = loss_ce
        # else:
        #     loss_metric = criterion_metric(features, labels)
        #     loss = loss_ce   + 0.1 * loss_metric
        # # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # 每5个epoch打印进度
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{config['n_epochs']}] Loss: {epoch_loss:.4f}")
train_time = time.time() - train_start
# 绘制损失曲线
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('loss_curve.png', bbox_inches='tight')
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
# 结果可视化
# =============================================
# 1. 文本报告
print("\n" + "=" * 50)
print(f"CNN分类结果（使用{len(selected_bands)}个SPA波段）")
print("=" * 50)
print(f"准确率: {accuracy_score(all_labels, all_preds):.4f}")
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


# 2. 混淆矩阵
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png')
    plt.show()


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, class_names)

print("\n" + "="*50)
print(f"\nSPA波段选择耗时: {spa_time:.2f} 秒")
print(f"\n模型训练总耗时: {train_time:.2f} 秒")
print(f"模型推理耗时: {test_time:.4f} 秒")
print(f"总建模时间 (训练+推理): {train_time + test_time:.2f} 秒")
print("="*50)