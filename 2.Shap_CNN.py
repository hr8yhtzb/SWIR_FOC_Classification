import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
import utils
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from net import SpectralDataset, SpectralCNN
import shap
import random
import net
import matplotlib.cm as cm
import torch.nn.functional as F
# 设置随机种子（与原始代码一致）
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================= 主程序 ==============================
# 数据加载（与原始代码一致）
data_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\5_CLASS\3.cleaned_data"
class_names = ["H", "A", "MI", "MO", "SE"]

AUG_PARAMS = {'noise_std': 0.008, 'random_state': 42}
cfg_1D_only = {'steps': ['1D']}

# cfg_msc_order = {
#         'steps': ['MSC'],
#         'MSC': True
#     }


# 加载数据
train_dataloader, test_dataloader = utils.load_hyper(
    data_dir, class_names, 0.6, 128,
    dynamic_aug=False, static_aug=True, augment_times=5,
    aug_params=AUG_PARAMS, preprocess_params=cfg_1D_only
)



# 提取数据
def extract_data_from_dataloader(dataloader):
    features, labels = [], []
    for x, y in dataloader:
        features.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


X_train, y_train = extract_data_from_dataloader(train_dataloader)
X_test, y_test = extract_data_from_dataloader(test_dataloader)
wavelengths = 1050 + np.arange(X_train.shape[1])

#
# XGBoots_selected_bands = [212, 63, 370, 81, 365, 117,
#                           541, 421, 369, 593, 213, 96,
#                           95, 99, 371, 328, 585, 571,
#                           367, 418, 428, 97, 568, 91,
#                           13, 443, 366, 422, 417, 329,
#                           364, 424, 598, 368, 207, 351,
#                           426, 373, 204, 416, 372, 361,
#                           360, 548, 195]
#
# XGBoots_selected_bands_MSC = [203, 285, 6, 458, 453, 4, 145,
#                               286, 5, 204, 354, 337, 558, 325,
#                               555, 321, 445, 412, 155, 150, 120,
#                               461, 284, 226, 149, 556, 227, 459,
#                               52, 454, 144, 179, 201, 514, 589,
#                               574, 148, 336, 205, 30, 260, 35,
#                               231, 365, 152]
#
selected_wavelengths_1 = [1539, 1153, 1168, 1535, 1587, 1243,
                        1359, 1285, 1117, 1604, 1555, 1261,
                        1187, 1366, 1460, 1547, 1404, 1514,
                        1445, 1347, 1343, 1321, 1330, 1362,
                        1290, 1151, 1181, 1512, 1319, 1348,
                        1101, 1115, 1394, 1425, 1125, 1416,
                        1489, 1457, 1272, 1266] # GA 40个波段

selected_wavelengths = sorted(selected_wavelengths_1)

selected_bands = [np.where(wavelengths == wl)[0][0] for wl in selected_wavelengths]

X_train_sel = X_train[:, selected_bands]
X_test_sel = X_test[:, selected_bands]

train_dataset = net.SpectralDataset(X_train_sel, y_train)
test_dataset = net.SpectralDataset(X_test_sel, y_test)

n_classes = len(class_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net.SpectralCNN_shap(n_classes=n_classes).to(device)

# 模型训练参数
criterion = nn.CrossEntropyLoss()


config = {
    'batch_size': 128,
    'lr': 1e-4,
    'n_epochs': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

optimizer = optim.Adam(model.parameters(), lr=config['lr'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

max_epoch = config['n_epochs']
train_losses = []
print("开始训练...")
for epoch in range(config['n_epochs']):
    # 训练阶段
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        # 前向传播
        outputs = model(inputs)
        loss_ce = criterion(outputs, labels)
        # loss_m = criterion_metric(features, labels)

        loss = loss_ce


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

print("\n开始 SHAP 波段重要性分析...")
# 确保模型在评估模式
model.eval()
model.cpu()

selected_wavelengths = wavelengths[selected_bands]

# 准备背景数据（随机抽取100个样本）
background = X_train_sel[np.random.choice(X_train.shape[0], 500, replace=False)]
background_tensor = torch.tensor(background, dtype=torch.float32)

# 创建SHAP解释器
explainer = shap.DeepExplainer(
    model,
    background_tensor.unsqueeze(1)  # 添加通道维度
)

# 选择测试样本（前50个样本）
test_samples = X_test_sel[:200]
test_tensor = torch.tensor(test_samples, dtype=torch.float32)

# 计算SHAP值
shap_values = explainer.shap_values(
    test_tensor.unsqueeze(1)  # 添加通道维度
)

# 波段重要性分析（绝对值平均）
shap_abs = np.abs(shap_values).mean(axis=0)  # 按类别和样本平均
band_importance = shap_abs.mean(axis=(0, 1))  # 按波段聚合

# 可视化波段重要性
plt.figure(figsize=(12, 6))
plt.plot(selected_wavelengths, band_importance, 'b-', linewidth=1)
plt.xlabel('Wavelength (nm)')
plt.ylabel('SHAP Value Magnitude')
plt.title('Band Importance Analysis using SHAP')
plt.grid(True)
plt.show()

# 可视化部分增强
plt.figure(figsize=(15, 10))

# 绘制所有类别的SHAP值（透明度叠加）
for i in range(n_classes):
    class_shap = np.abs(shap_values[i]).mean(axis=(0, 1))  # 按样本和通道平均
    plt.plot(selected_wavelengths, class_shap, alpha=0.6,
             label=f'Class {class_names[i]}')

# 绘制总体重要性曲线
overall_importance = np.abs(shap_values).mean(axis=(0, 1, 2))
plt.plot(selected_wavelengths, overall_importance, 'k--', linewidth=2,
         label='Overall Importance')

plt.title('SHAP Value Distribution by Wavelength and Class', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Mean |SHAP Value|', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 打印前10重要波段
top_n = 6
sorted_idx = np.argsort(overall_importance)[-top_n:][::-1]
top_bands = selected_wavelengths[sorted_idx].astype(int)
top_values = overall_importance[sorted_idx]

print("\nTop 10 Important Wavelengths:")
print(f"{'Rank':<5} | {'Wavelength (nm)':<15} | {'Importance':<10}")
print("-"*40)
for i, (band, val) in enumerate(zip(top_bands, top_values)):
    print(f"{i+1:<5} | {band:<15} | {val:.4f}")


# 提取前10个重要波段的数据
X_selected = test_samples[:, sorted_idx]  # 筛选重要波段
wavelengths_selected = selected_wavelengths[sorted_idx]  # 对应的波长值

# 创建对应的SHAP值（注意保持与筛选后的波段对应）
shap_values_selected = np.array(shap_values)[:, :, :, sorted_idx]  # 筛选对应波段
shap_values_reshaped = np.squeeze(shap_values_selected, axis=2)
# 转换为适合summary_plot的格式（合并多类别）
shap_values_combined = np.mean(shap_values_reshaped, axis=0)# 按类别平均

# 验证最终维度
print(f"SHAP值矩阵形状: {shap_values_combined.shape}")
print(f"特征矩阵形状: {X_selected.shape}")

# 1. 条形摘要图（显示特征重要性）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_combined,
                 X_selected,
                 feature_names=wavelengths_selected.astype(int),
                 plot_type="bar",
                 max_display=10,
                 show=False)
plt.title("Top 10 Important Bands - Mean SHAP Value")
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# 数据预处理：提取每个类别的平均|SHAP值|
class_shap_importance = []
for class_idx in range(n_classes):
    # 计算单个类别的平均绝对值 (50 samples → mean)
    importance = np.abs(shap_values_reshaped[class_idx]).mean(axis=0)  # shape (10,)
    class_shap_importance.append(importance)

# 转换为DataFrame便于绘图
import pandas as pd
df_importance = pd.DataFrame(
    data=np.array(class_shap_importance).T,  # 转置使列为类别
    index=wavelengths_selected.astype(int),
    columns=class_names
)
df_importance.index.name = 'Wavelength(nm)'


# 可视化分层条形图
plt.figure(figsize=(14, 8))
df_importance.plot(kind='barh',
                   colormap='viridis',
                   width=0.85,
                   edgecolor='w',
                   linewidth=0.8)
plt.gca().invert_yaxis()  # 按重要性从高到低排序
# plt.title('Band Importance Across Different Classes', fontsize=14)
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.ylabel('Wavelength (nm)', fontsize=12)
plt.legend(title='Class', loc='lower right')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('Band Importance Across Different Classes.pdf', dpi=800, bbox_inches='tight')
plt.show()


# 计算排序索引（按总体重要性）
overall_rank = df_importance.sum(axis=1).sort_values(ascending=False).index
df_sorted = df_importance.loc[overall_rank]

plt.figure(figsize=(14, 8))  # 适当增加画布高度
ax = df_sorted.plot(
    kind='bar',
    stacked=True,
    colormap='tab20',
    edgecolor='w',
    width=0.85
)

# 优化布局参数
# plt.subplots_adjust(right=0.65)  # 右侧保留35%空白

# plt.title('Stacked Band Importance by Class', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Cumulative SHAP Impact', fontsize=12)
plt.xticks(rotation=0, ha='center', fontsize=12)
# plt.yticks(fontsize=10)

# 优化图例参数
legend = plt.legend(
    title='Class',
    # bbox_to_anchor=(0.8, 0.95),  # 相对于axes的位置
    loc='upper right',
    frameon=True,
    fontsize=12,
    title_fontsize=12
)

# 设置图例样式
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#DDDDDD')
legend.get_frame().set_linewidth(0.8)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# 先保存再显示
plt.savefig('5_stacked_importance.pdf', dpi=800, bbox_inches='tight')
plt.show()

# 2. 散点摘要图（显示特征值-SHAP关系）
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_combined,
                 X_selected,
                 feature_names=wavelengths_selected.astype(int),
                 plot_type="dot",
                 max_display=10,
                 color=plt.get_cmap('viridis'),
                 show=False)
plt.title("SHAP Value Distribution by Band Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 设置全局样式
plt.style.use('seaborn-darkgrid')

# 循环绘制每个类别的散点摘要图
for class_idx in range(n_classes):
    # 创建独立画布
    plt.figure(figsize=(14, 8))

    # 生成散点摘要图
    shap.summary_plot(
        shap_values=shap_values_reshaped[class_idx],  # 当前类别的SHAP值 (50,10)
        features=X_selected,  # 特征矩阵 (50,10)
        feature_names=[f"{w}nm" for w in wavelengths_selected.astype(int)],
        plot_type="dot",  # 指定散点模式
        color=cm.get_cmap('tab10')(class_idx),  # 按类别分配主色调
        alpha=0.7,  # 点透明度
        show=False
    )

    # 增强可视化元素
    plt.title(f"Class '{class_names[class_idx]}': Feature Values vs SHAP Values",
              fontsize=14, pad=20)
    plt.xlabel("Normalized Band Value", fontsize=12, labelpad=10)
    plt.ylabel("SHAP Value (Impact on Prediction)", fontsize=12, labelpad=10)

    # 添加辅助线
    plt.axhline(y=0, color='#444444', linestyle='--', linewidth=0.8, zorder=0)

    # 调整颜色条
    cb = plt.gcf().axes[-1]
    cb.set_ylabel('Band Value Magnitude', rotation=270, labelpad=15)

    # 优化布局
    plt.tight_layout()
    plt.savefig(f'Shap_class_{class_names[class_idx]}_shap.png', dpi=800, bbox_inches='tight')
    plt.show()
