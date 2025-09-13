import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import utils
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import matplotlib.pyplot as plt
from net import SpectralDataset
import random
import net
import matplotlib.cm as cm
from tqdm import tqdm

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================= 随机蛙跳算法 ==============================
class RandomFrogSelector:
    def __init__(self, model_class, X, y, wavelengths, device,
                 n_iter=200, init_ratio=0.2, Q=0.3, cv=3,
                 n_epochs=50, batch_size=64, lr=1e-4):
        self.model_class = model_class
        self.X = X
        self.y = y
        self.wavelengths = wavelengths
        self.device = device
        self.n_iter = n_iter
        self.init_ratio = init_ratio
        self.Q = Q
        self.cv = cv
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.n_bands = X.shape[1]
        self.prob = np.ones(self.n_bands) / self.n_bands
        self.selected_history = []
        self.acc_history = []

    def _evaluate_subset(self, subset):
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=seed)
        acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X)):
            # 数据准备
            train_set = SpectralDataset(self.X[train_idx][:, subset], self.y[train_idx])
            val_set = SpectralDataset(self.X[val_idx][:, subset], self.y[val_idx])

            # 模型初始化
            model = self.model_class(n_classes=len(class_names)).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            # 简化训练过程
            best_acc = 0.0
            for epoch in range(self.n_epochs):
                model.train()
                for inputs, labels in DataLoader(train_set, batch_size=self.batch_size, shuffle=True):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    _, outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # 验证
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for inputs, labels in DataLoader(val_set, batch_size=self.batch_size):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        _, outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                acc = correct / total
                if acc > best_acc:
                    best_acc = acc

            acc_scores.append(best_acc)

        return np.mean(acc_scores)

    def _update_probability(self, selected, acc):
        delta = np.zeros_like(self.prob)
        delta[selected] = acc
        self.prob = (1 - self.Q) * self.prob + self.Q * delta
        self.prob = np.clip(self.prob, 1e-6, 1)  # 数值稳定性处理
        self.prob /= self.prob.sum()

    def select_bands(self, top_n=50):
        # 初始子集
        init_size = max(2, int(self.n_bands * self.init_ratio))
        selected = np.random.choice(self.n_bands, init_size, replace=False, p=self.prob)

        # 主循环
        pbar = tqdm(range(self.n_iter), desc="Random Frog 波段选择")
        for _ in pbar:
            # 生成候选子集
            candidate = np.random.choice(self.n_bands, size=len(selected), replace=False, p=self.prob)

            # 评估候选子集
            acc = self._evaluate_subset(candidate)

            # 概率更新
            self._update_probability(candidate, acc)

            # 记录历史
            self.selected_history.append(candidate)
            self.acc_history.append(acc)

            # 动态显示信息
            pbar.set_postfix_str(f"当前准确率: {acc:.4f} | 波段数: {len(candidate)}")

        # 最终选择
        self.final_selection = np.argsort(-self.prob)[:top_n]
        return self.final_selection

    def visualize_results(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.wavelengths, self.prob, 'b-', alpha=0.7, label='选择概率')
        plt.scatter(self.wavelengths[self.final_selection],
                    self.prob[self.final_selection],
                    c='red', s=50, edgecolors='k', label='选中波段')

        plt.title("随机蛙跳波段选择结果", fontsize=14)
        plt.xlabel("波长 (nm)", fontsize=12)
        plt.ylabel("选择概率", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()




# ============================= 数据加载 ==============================
data_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\5_CLASS\3.cleaned_data"
class_names = ["H", "A", "MI", "MO", "SE"]

AUG_PARAMS = {'noise_std': 0.008, 'random_state': 42}
cfg_1D_order = {'steps': ['1D']}

# 加载并预处理数据
train_dataloader, test_dataloader = utils.load_hyper(
    data_dir, class_names, 0.6, 32,
    dynamic_aug=False, static_aug=True, augment_times=5,
    aug_params=AUG_PARAMS, preprocess_params=cfg_1D_order
)

def extract_data(dataloader):
    features, labels = [], []
    for x, y in dataloader:
        features.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)

X_train, y_train = extract_data(train_dataloader)
X_test, y_test = extract_data(test_dataloader)
wavelengths = 1050 + np.arange(X_train.shape[1])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
selector = RandomFrogSelector(
        model_class=net.SpectralCNN,
        X=X_train,
        y=y_train,
        wavelengths=wavelengths,
        device=device,
        n_iter=500,       # 迭代次数（实际使用建议500+）
        init_ratio=0.1,
        Q=0.3,
        cv=3,
        n_epochs=30,      # 快速验证轮次
        batch_size=64,
        lr=1e-4
    )

# 执行波段选择
print("\n开始随机蛙跳波段选择...")
selected_bands = selector.select_bands(top_n=40)

# 可视化结果
selector.visualize_results()

# 打印选中的波段
print(list(selected_bands))
print(list(1050 + selected_bands))
print("Selected bands:", selected_bands + 1050)


X_train_sel = X_train[:, selected_bands]
X_test_sel = X_test[:, selected_bands]

# 可视化选择结果
plt.figure(figsize=(12, 5))
plt.plot(wavelengths, X_train.mean(axis=0), alpha=0.6, label='平均光谱')
plt.scatter(wavelengths[selected_bands],
            X_train.mean(axis=0)[selected_bands],
            c='red', s=50, edgecolor='k', zorder=3, label='手动选择波段')

config = {
    'batch_size': 64,
    'lr': 1e-4,
    'n_epochs': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

train_dataset = net.SpectralDataset(X_train_sel, y_train)
test_dataset = net.SpectralDataset(X_test_sel, y_test)

# 初始化模型
n_bands = len(selected_bands)
n_classes = len(class_names)

# model = net.SpectralCNN_2(n_bands, n_classes).to(config['device'])
model = net.SpectralCNN(n_classes).to(config['device'])

criterion = nn.CrossEntropyLoss()
# criterion_metric = EuclideanMetricLoss_pro(margin=5.0).cuda()

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

    for inputs, labels in train_loader:
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        # 前向传播
        features, outputs = model(inputs)
        loss_ce = criterion(outputs, labels)

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
# torch.save(model.state_dict(), 'spectral_cnn_model_metric.pth')
# # 保存预处理后的数据
# np.savez('preprocessed_data.npz',
#          X_train=X_train_sel,
#          X_test=X_test_sel,
#          y_train=y_train,
#          y_test=y_test)
# # 保存类别名称
# with open('class_names.txt', 'w') as f:
#     f.write('\n'.join(class_names))

# from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
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