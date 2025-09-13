import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import utils
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from net import SpectralDataset
import random
import net
from tqdm import tqdm
from collections import deque

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


# ============================= 遗传算法波段选择 ==============================
class GASelector:
    def __init__(self, model_class, X, y, wavelengths, device,
                 pop_size=50, n_generations=100, elite_ratio=0.1,
                 crossover_prob=0.8, mutation_prob=0.2,
                 cv=3, n_epochs=30, batch_size=64, lr=1e-4):

        self.model_class = model_class
        self.X = X
        self.y = y
        self.wavelengths = wavelengths
        self.device = device

        # GA参数
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.elite_size = int(pop_size * elite_ratio)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # 训练参数
        self.cv = cv
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.n_bands = X.shape[1]
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        self.population = []
        self.selected_history = []

    def _initialize_population(self, chromosome_length, max_bands=60):
        """初始化种群，每个个体包含约max_bands个波段"""
        return [np.random.choice(chromosome_length, max_bands, replace=False)
                for _ in range(self.pop_size)]

    def _evaluate_subset(self, subset):
        """评估子集适应度（与随机蛙跳相同的评估逻辑）"""
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=seed)
        acc_scores = []

        for train_idx, val_idx in kfold.split(self.X):
            # 数据准备
            train_set = SpectralDataset(self.X[train_idx][:, subset], self.y[train_idx])
            val_set = SpectralDataset(self.X[val_idx][:, subset], self.y[val_idx])

            # 模型初始化
            model = self.model_class(n_classes=len(class_names)).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            # 训练过程
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

    def _crossover(self, parent1, parent2):
        """均匀交叉"""
        child = np.unique(np.concatenate([parent1, parent2]))
        return np.random.choice(child, size=len(parent1), replace=False)

    def _mutate(self, individual):
        """变异操作：随机替换部分波段"""
        mutate_num = max(1, int(len(individual) * 0.1))
        preserved = np.random.choice(individual, len(individual) - mutate_num, replace=False)
        new_bands = np.random.choice(self.n_bands, mutate_num, replace=False)
        return np.unique(np.concatenate([preserved, new_bands]))

    def _select_parents(self, fitness_scores):
        """轮盘赌选择"""
        probs = fitness_scores / fitness_scores.sum()
        return self.population[np.random.choice(len(self.population), p=probs)]

    def select_bands(self, top_n=60):
        """主遗传算法流程"""
        self.population = self._initialize_population(self.n_bands)

        # 使用自定义格式的进度条
        with tqdm(total=self.n_generations, desc="GA波段选择",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for gen in range(self.n_generations):
                fitness = []
                evaluated_count = 0  # 记录实际评估次数

                # 带评估进度的嵌套进度条
                with tqdm(self.population, desc=f"第{gen + 1}代评估", leave=False) as inner_pbar:
                    for ind in inner_pbar:
                        if tuple(ind) in self.selected_history:
                            fitness.append(0)
                        else:
                            score = self._evaluate_subset(ind)
                            fitness.append(score)
                            self.selected_history.append(tuple(ind))
                            evaluated_count += 1
                        inner_pbar.set_postfix_str(f"新评估:{evaluated_count}")

                # 更新逻辑保持不变...
                max_fitness = np.max(fitness)
                if max_fitness > self.best_fitness:
                    self.best_fitness = max_fitness
                    self.best_individual = self.population[np.argmax(fitness)]

                self.fitness_history.append(max_fitness)

                # 更新主进度条
                pbar.update(1)
                pbar.set_postfix_str(f"最佳:{self.best_fitness:.4f}",
                                     f"波段数:{len(self.best_individual)}")

        return self.best_individual[:top_n]

    def visualize_results(self):
        """可视化进化过程"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.fitness_history, 'b-', alpha=0.7)
        plt.title("遗传算法进化过程")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.grid(True, alpha=0.3)
        plt.show()


# ============================= 数据加载（保持不变） ==============================
data_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\5_CLASS\3.cleaned_data"
class_names = ["H", "A", "MI", "MO", "SE"]

AUG_PARAMS = {'noise_std': 0.008, 'random_state': 42}
cfg_1D_order = {'steps': ['1D']}

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

# ============================= 使用遗传算法 ==============================
selector = GASelector(
    model_class=net.SpectralCNN,
    X=X_train,
    y=y_train,
    wavelengths=wavelengths,
    device=device,
    pop_size=30,  # 种群大小
    n_generations=50,  # 进化代数
    elite_ratio=0.1,  # 精英保留比例
    crossover_prob=0.8,  # 交叉概率
    mutation_prob=0.3,  # 变异概率
    cv=3,
    n_epochs=30,
    batch_size=64,
    lr=1e-4
)

print("\n开始遗传算法波段选择...")
selected_bands = selector.select_bands(top_n=6)
selector.visualize_results()

# 后续建模流程保持不变...
# （此处保留原有建模、训练和评估代码）

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