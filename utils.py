import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import pywt
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.manifold import TSNE
# import shap
from captum.attr import DeepLift


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, steps=None, **step_params):
        """
        参数说明：
        - steps: 处理步骤顺序列表，例如 ['MSC', 'SNV', 'SG']
        - step_params: 各步骤的参数配置
        """
        self.steps = steps or []
        self.step_params = step_params
        self.fitted_params_ = {}  # 存储需要保存的参数

    def fit(self, X, y=None):
        """仅对需要拟合的步骤进行计算"""
        for step in self.steps:
            if step == 'MSC':
                self._fit_msc(X)
            elif step == 'SNV':
                pass  # SNV无需保存参数
            elif step == 'SG':
                self._fit_sg(X)
            elif step == 'DeTrend':
                self._fit_detrend(X)
            # 其他步骤处理...
        return self

    def transform(self, X):
        """按顺序应用所有处理步骤"""
        processed = X.copy()
        for step in self.steps:
            if step == 'MSC':
                processed = self._apply_msc(processed)
            elif step == 'SNV':
                processed = self._apply_snv(processed)
            elif step == 'SG':
                processed = self._apply_sg(processed)
            elif step == '1D':
                processed = self._apply_deriv(processed, order=1)
            elif step == '2D':
                processed = self._apply_deriv(processed, order=2)
            elif step == 'Wavelet':
                processed = self._apply_wavelet(processed)
            elif step == 'DeTrend':
                processed = self._apply_detrend(processed)

        return processed

    def fit_transform(self, X, y=None):
        """整合fit和transform流程"""
        self.fit(X)
        return self.transform(X)

    # --------------------------
    # 各步骤具体实现
    # --------------------------
    def _fit_detrend(self, X):
        """获取去趋势化参数（仅存储配置）"""
        params = self.step_params.get('DeTrend', {})
        self.fitted_params_['detrend_order'] = params.get('poly_order', 1)

    def _apply_detrend(self, X):
        """应用去趋势化处理"""
        order = self.fitted_params_.get('detrend_order', 1)

        x_axis = np.arange(X.shape[1])  # 假设波长为等间距序列
        detrended = np.zeros_like(X)

        for i in range(X.shape[0]):
            # 多项式拟合（degree=order）
            coeffs = np.polyfit(x_axis, X[i], deg=order)
            trend = np.polyval(coeffs, x_axis)
            detrended[i] = X[i] - trend

        return detrended
    def _fit_msc(self, X):
        """MSC参数拟合"""
        self.fitted_params_['msc_ref'] = np.mean(X, axis=0)

    def _apply_msc(self, X):
        """应用MSC校正"""
        ref = self.fitted_params_.get('msc_ref', None)
        if ref is None:
            return X

        corrected = np.zeros_like(X)
        for i in range(len(X)):
            lr = LinearRegression()
            lr.fit(ref.reshape(-1, 1), X[i].reshape(-1, 1))
            corrected[i] = (X[i] - lr.intercept_) / lr.coef_[0]
        return corrected

    def _fit_sg(self, X):
        """SG滤波器参数校验"""
        params = self.step_params.get('SG', {})
        self.fitted_params_['sg_window'] = params.get('window', 21)
        self.fitted_params_['sg_poly'] = params.get('poly_order', 3)

    def _apply_sg(self, X):
        """应用SG滤波"""
        window = self.fitted_params_.get('sg_window', 21)
        poly_order = self.fitted_params_.get('sg_poly', 3)
        return savgol_filter(X, window, poly_order, deriv=0)

    def _apply_snv(self, X):
        """应用SNV（无需保存参数）"""
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    def _apply_deriv(self, X, order):
        """导数计算"""
        delta = 1  # 假设波长等间距
        deriv = np.zeros_like(X)
        if order == 1:
            deriv[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / (2 * delta)
            deriv[:, 0] = (X[:, 1] - X[:, 0]) / delta
            deriv[:, -1] = (X[:, -1] - X[:, -2]) / delta
        elif order == 2:
            deriv[:, 1:-1] = (X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]) / (delta ** 2)
            deriv[:, 0] = (X[:, 2] - 2 * X[:, 1] + X[:, 0]) / (delta ** 2)
            deriv[:, -1] = (X[:, -1] - 2 * X[:, -2] + X[:, -3]) / (delta ** 2)
        return deriv

    def _apply_wavelet(self, X):
        """小波去噪实现"""
        params = self.step_params.get('Wavelet', {})
        wavelet = params.get('wavelet', 'sym10')
        level = params.get('level', 3)

        def denoise(signal):
            coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
            coeffs = [pywt.threshold(c, uthresh, mode='soft') if i > 0 else c
                      for i, c in enumerate(coeffs)]
            return pywt.waverec(coeffs, wavelet, mode='periodization')[:len(signal)]

        return np.apply_along_axis(denoise, 1, X)


class DiseaseOrientedAugmentation:
    """面向香蕉枯萎病的增强处理器"""

    def __init__(self, wavelengths,
                 noise_std=0.005,
                 warp_scale=0.08,
                 mask_ratio=0.15,
                 key_bands=[1200, 1350, 1450],
                 random_state=None
                 ):
        self.wavelengths = wavelengths
        self.noise_std = noise_std
        self.warp_scale = warp_scale
        self.mask_ratio = mask_ratio
        self.key_indices = [np.argmin(np.abs(wavelengths - b)) for b in key_bands]
        print(f"关键波段保护索引：{self.key_indices}")

    def _add_noise(self, spectrum):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, size=spectrum.shape)
        return spectrum + noise

    def _spectral_warp(self, spectrum):
        """光谱非线性扭曲"""
        n_bands = len(spectrum)
        x = np.linspace(0, 1, n_bands)
        warp = CubicSpline(x, self.warp_scale * np.random.randn(n_bands))(x)
        return spectrum * (1 + warp)
    def _local_shift(self, spectrum):
        """局部波段偏移（模拟叶片形变）"""
        shift_window = 10  # 在20nm窗口内偏移
        shift = np.random.randint(-3, 3)  # ±3个波段
        start = np.random.randint(0, len(spectrum) - shift_window)
        spectrum[start:start + shift_window] = np.roll(
            spectrum[start:start + shift_window], shift)
        return spectrum

    # 上述局部偏移实现

    def _safe_mask(self, spectrum):
        """避开关键特征区的随机遮挡"""
        mask = np.ones_like(spectrum)

        # 构建保护区域（关键波段±10个索引）
        protected_zones = []
        for idx in self.key_indices:
            protected_zones.extend(range(max(0, idx - 10), min(len(spectrum), idx + 10)))

        # 生成随机遮挡（非保护区域）
        rand_mask = np.random.rand(len(spectrum)) > self.mask_ratio
        for i in range(len(spectrum)):
            if i not in protected_zones:
                mask[i] = rand_mask[i]

        return spectrum * mask

    # 上述安全遮挡实现

    def _simulate_water_stress(self, spectrum):
        """基于索引的含水量特征增强"""
        # 找到1450nm对应的索引
        water_idx = self.key_indices[2]  # 假设key_bands第三个是1450

        # 在索引附近施加衰减
        start = max(0, water_idx - 5)
        end = min(len(spectrum), water_idx + 5)
        attenuation = 1 - np.random.uniform(0.1, 0.3)
        spectrum[start:end] *= attenuation

        return spectrum

    # 上述含水量模拟实现

    def __call__(self, spectrum):
        functions = [
            self._add_noise,
            self._local_shift,
            self._spectral_warp,
            self._safe_mask,
            self._simulate_water_stress
        ]
        selected = np.random.choice(functions, 3, replace=False)
        for func in selected:
            spectrum = func(spectrum)
        return spectrum

    def augment_batch(self, X_batch):
        """批量增强方法"""
        return np.array([self(sample) for sample in X_batch])

class SpectralAugmentation:
    """光谱数据增强处理器"""

    def __init__(self,
                 noise_std=0.01,
                 warp_scale=0.1,
                 random_state = None):
        """
        Args:
            noise_std (float): 高斯噪声标准差 (默认0.01)
            shift_max (int): 最大平移点数 (默认5)
            warp_scale (float): 光谱扭曲强度 (默认0.1)
            mask_ratio (float): 随机遮挡比例 (默认0.1)
        """
        self.rng = np.random.RandomState(random_state)
        self.noise_std = noise_std
        self.warp_scale = warp_scale

    def add_noise(self, spectrum):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, size=spectrum.shape)
        return spectrum + noise

    def random_shift(self, spectrum):
        """局部波段偏移（模拟叶片形变）"""
        shift_window = 10  # 在20nm窗口内偏移
        shift = np.random.randint(-3, 3)  # ±3个波段
        start = np.random.randint(0, len(spectrum) - shift_window)
        spectrum[start:start + shift_window] = np.roll(
            spectrum[start:start + shift_window], shift)
        return spectrum

    def spectral_warp(self, spectrum):
        """光谱非线性扭曲"""
        n_bands = len(spectrum)
        x = np.linspace(0, 1, n_bands)
        warp = CubicSpline(x, self.warp_scale * np.random.randn(n_bands))(x)
        return spectrum * (1 + warp)

    def __call__(self, spectrum):
        """顺序应用所有增强"""
        functions = [
            self.add_noise
        ]
        # 随机选择3种增强组合
        selected = np.random.choice(functions, 1, replace=False)
        for func in selected:
            spectrum = func(spectrum)
        return spectrum

    def augment_batch(self, X_batch):
        """批量增强方法"""
        return np.array([self(sample) for sample in X_batch])


class AugmentedDataset(Dataset):
    """支持数据增强的自定义数据集"""

    def __init__(self, data_tensor, label_tensor, augment=None):
        self.data = data_tensor
        self.labels = label_tensor
        self.augment = augment  # 增强处理器实例

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment:
            # 转换为numpy进行增强
            x_np = x.numpy()
            x_aug = self.augment(x_np)
            x = torch.from_numpy(x_aug).float()

        return x, y


def expand_with_augmentation(original_data, original_labels, augmentor, augment_times=3):
    """静态数据增强扩展函数
    Args:
        original_data (np.ndarray): 原始数据矩阵，形状 (n_samples, n_bands)
        original_labels (np.ndarray): 原始标签向量，形状 (n_samples,)
        augmentor (SpectralAugmentation): 数据增强处理器实例
        augment_times (int): 每个样本的增强次数
    Returns:
        augmented_data (np.ndarray): 增强后的数据矩阵
        augmented_labels (np.ndarray): 增强后的标签向量
    """
    augmented_list = []
    label_list = []

    # 保留原始数据
    augmented_list.append(original_data)
    label_list.append(original_labels)

    # 生成增强数据
    for _ in range(augment_times):
        batch_aug = np.array([augmentor(sample) for sample in original_data])
        augmented_list.append(batch_aug)
        label_list.append(original_labels)  # 标签与原始数据相同

    # 合并数据
    augmented_data = np.vstack(augmented_list)
    augmented_labels = np.concatenate(label_list)

    return augmented_data, augmented_labels

def loadData(data_dir, class_names):
    """
    加载高光谱数据
    Args:
        data_dir (str): 存放四个Excel文件的目录路径
    Returns:
        data (np.array): 光谱数据数组 (n_samples, n_bands)
        labels (np.array): 对应标签数组 (n_samples,)
    """

    label_map = {name: idx for idx, name in enumerate(class_names)}  # 类别到标签的映射

    # 初始化存储
    all_data = []
    all_labels = []

    # 遍历每个类别文件
    for class_name in class_names:
        # 构建文件路径
        file_path = os.path.join(data_dir, f"{class_name}.xlsx")

        # 读取Excel文件
        df = pd.read_excel(file_path, index_col=0, header=0)

        # 转置数据：行变样本，列变波段
        samples = df.T.values  # (n_samples, n_bands)

        # 生成对应标签
        labels = np.full(len(samples), label_map[class_name])

        # 收集数据
        all_data.append(samples)
        all_labels.append(labels)

    # 合并所有数据
    data = np.concatenate(all_data, axis=0)  # (n_total_samples, n_bands)
    labels = np.concatenate(all_labels, axis=0)  # (n_total_samples,)

    print(f"光谱数据维度: {data.shape}")  # (n_samples, n_bands)
    print(f"标签维度: {labels.shape}")

    return data, labels


def split_data_fix(data, labels, tr_percent, rand_state=None):
    train_set_size = [tr_percent] * len(np.unique(labels))
    # 调用 split_data 函数
    return split_data(data, labels, 0, train_set_size, rand_state)

def split_data(data, labels, tr_percent, train_set_size=None, rand_state=None):

    # 获取类别信息和每类样本数
    unique_labels, class_counts = np.unique(labels, return_counts=True)

    # 初始化存储
    train_x, train_y = [], []
    test_x, test_y = [], []

    # 设置随机种子
    rng = np.random.RandomState(rand_state)

    # 遍历每个类别
    for i, cl in enumerate(unique_labels):
        # 获取当前类别的样本索引
        bind = np.where(labels == cl)[0]
        rng.shuffle(bind)  # 随机打乱

        # 计算训练集样本数
        if train_set_size is not None and len(train_set_size) == len(unique_labels):
            n_train = train_set_size[i]  # 按固定样本数划分
        else:
            if tr_percent < 1:
                n_train = int(class_counts[i] * tr_percent)  # 按比例划分
            else:
                n_train = min(tr_percent, class_counts[i])  # 按固定样本数划分

        # 划分训练集和测试集
        train_indices = bind[:n_train]
        test_indices = bind[n_train:]

        # 收集数据
        train_x.append(data[train_indices])
        train_y.append(labels[train_indices])
        test_x.append(data[test_indices])
        test_y.append(labels[test_indices])

    # 合并数据
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    print(f"训练集样本数: {len(train_x)}")
    print(f"测试集样本数: {len(test_x)}")

    return train_x, test_x, train_y, test_y


def create_dataloaders(train_x, train_y, test_x, test_y, batch_size=32, shuffle=True, num_workers=4):
    """
    创建训练和测试 DataLoader
    Args:
        train_x (np.array): 训练集数据
        train_y (np.array): 训练集标签
        test_x (np.array): 测试集数据
        test_y (np.array): 测试集标签
        batch_size (int): 批量大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载的线程数
    Returns:
        train_dataloader (DataLoader): 训练集 DataLoader
        test_dataloader (DataLoader): 测试集 DataLoader
    """
    # 将 NumPy 数组转换为 PyTorch 张量
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)

    # 创建 TensorDataset
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集通常不需要打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_dataloader, test_dataloader


#

def load_hyper_val(data_dir, class_names, tr_percent, batch_size,
               dynamic_aug=False, static_aug=False, augment_times=3,
               aug_params=None, preprocess_params=None):
    # 加载原始数据
    data, labels = loadData(data_dir, class_names)

    # 新增的三级划分逻辑
    if tr_percent < 1:
        # 第一次划分：训练集60%，临时测试集40%
        train_x_raw, temp_x, train_y, temp_y = train_test_split(
            data, labels,
            test_size=(1-tr_percent),  # 保留40%用于验证和测试
            stratify=labels,
            random_state=42
        )

        # 第二次划分：将40%分为验证和测试各20%
        val_x_raw, test_x_raw, val_y, test_y = train_test_split(
            temp_x, temp_y,
            test_size=0.5,  # 各取50%即总体的20%
            stratify=temp_y,
            random_state=42
        )
    else:
        # 固定样本数划分模式需要调整逻辑
        raise NotImplementedError("固定样本数模式需重新实现三级划分")

    # ------------------
    # 预处理阶段（新增验证集处理）
    # ------------------
    if preprocess_params is not None:
        preprocessor = Preprocessor(**preprocess_params)
        # 训练集拟合并转换
        train_x = preprocessor.fit_transform(train_x_raw)
        # 验证集和测试集仅转换
        val_x = preprocessor.transform(val_x_raw)
        test_x = preprocessor.transform(test_x_raw)
    else:
        train_x = train_x_raw
        val_x = val_x_raw
        test_x = test_x_raw

    # ------------------
    # 数据增强（仅训练集）
    # ------------------
    if static_aug and augment_times > 0:
        augmentor = SpectralAugmentation(**(aug_params or {}))
        train_x, train_y = expand_with_augmentation(
            train_x, train_y, augmentor=augmentor, augment_times=augment_times)

    # ------------------
    # 数据保存（新增验证集保存）
    # ------------------
    save_dir = os.path.join(data_dir, "processed_by_class")
    os.makedirs(save_dir, exist_ok=True)

    # 保存函数封装
    def save_processed_data(x_data, y_data, suffix):
        for class_idx, class_name in enumerate(class_names):
            mask = (y_data == class_idx)
            class_data = x_data[mask]

            df = pd.DataFrame(class_data.T)
            df.insert(0, "Band", band_names)

            file_path = os.path.join(save_dir, f"{class_name}_{suffix}.xlsx")
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, startrow=1)
                worksheet = writer.sheets['Sheet1']
                for col in range(1, df.shape[1]):
                    worksheet.cell(row=1, column=col + 1, value=1)
                worksheet.cell(row=1, column=1, value="Band")

    # 获取波段信息
    n_bands = train_x.shape[1]
    band_names = list(range(1050, 1050 + n_bands))

    # 分别保存三个数据集
    save_processed_data(train_x, train_y, "train")
    save_processed_data(val_x, val_y, "val")
    save_processed_data(test_x, test_y, "test")

    # ------------------
    # 转换为Tensor（新增验证集）
    # ------------------
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)

    val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
    val_y_tensor = torch.tensor(val_y, dtype=torch.long)

    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)

    # ------------------
    # 数据集构建（新增验证集）
    # ------------------
    if dynamic_aug:
        aug = SpectralAugmentation(**(aug_params or {}))
        train_dataset = AugmentedDataset(train_x_tensor, train_y_tensor, augment=aug)
    else:
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)

    val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

    # ------------------
    # DataLoader配置（新增验证集）
    # ------------------
    tr_batch_size = min(batch_size, len(train_x))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=tr_batch_size,
        shuffle=True,  # 仅训练集需要shuffle
        pin_memory=torch.cuda.is_available()
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # 新增信息输出
    print(f"[预处理+增强后] 训练样本数: {len(train_dataset)}")
    print(f"[预处理后] 验证样本数: {len(val_dataset)}")
    print(f"[预处理后] 测试样本数: {len(test_dataset)}")

    return train_dataloader, val_dataloader, test_dataloader


def load_hyper(data_dir, class_names, tr_percent, batch_size,
               dynamic_aug=False, static_aug=False, augment_times=3,
               aug_params=None, preprocess_params=None):
    # 加载原始数据
    data, labels = loadData(data_dir, class_names)

    # 划分训练测试集（保持原始状态）
    if tr_percent < 1:
        train_x_raw, test_x_raw, train_y, test_y = train_test_split(
            data, labels, test_size=1 - tr_percent, stratify=labels, random_state=42)
    elif tr_percent==1:
        train_x_raw = data
        train_y = labels
        test_x_raw = np.zeros(0)  # 创建空数组
        test_y = np.zeros(0)
        preprocessor = Preprocessor(**preprocess_params)
        # 训练集拟合并转换
        train_x = preprocessor.fit_transform(train_x_raw)
        if static_aug and augment_times > 0:
            augmentor = SpectralAugmentation(**(aug_params or {}))
            train_x, train_y = expand_with_augmentation(
                train_x, train_y, augmentor=augmentor, augment_times=augment_times)
        train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
        train_y_tensor = torch.tensor(train_y, dtype=torch.long)
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        tr_batch_size = min(batch_size, len(train_x))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=tr_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        return train_dataloader, 1

    else:
        train_x_raw, test_x_raw, train_y, test_y = split_data_fix(data, labels, tr_percent)

    # ------------------
    # 预处理阶段
    # ------------------
    if preprocess_params is not None:
        preprocessor = Preprocessor(**preprocess_params)
        # 训练集拟合并转换
        train_x = preprocessor.fit_transform(train_x_raw)
        # 测试集仅转换
        test_x = preprocessor.transform(test_x_raw)
    else:
        train_x = train_x_raw
        test_x = test_x_raw


    # MSC合理性
    # msc_diff = train_x - train_x_raw  # 计算MSC处理差异
    # print(f"MSC处理后健康样本平均变异度: {np.mean(msc_diff[train_y == 0])}")  # 显著高于其他类别时说明处理失真

    # ------------------
    # 数据增强阶段
    # ------------------
    if static_aug and augment_times > 0:
        augmentor = SpectralAugmentation(**(aug_params or {}))
        train_x, train_y = expand_with_augmentation(
            train_x, train_y, augmentor=augmentor, augment_times=augment_times)

    save_dir = os.path.join(data_dir, "processed_by_class")
    os.makedirs(save_dir, exist_ok=True)

    # 获取波段名称（假设原始数据列名为1050-1650）
    n_bands = train_x.shape[1]
    band_names = list(range(1050, 1050 + n_bands)) # 动态生成波段名称

    # 按类别分割并保存
    for class_idx, class_name in enumerate(class_names):
        # 提取当前类别的数据
        mask = (train_y == class_idx)
        class_data = train_x[mask]  # (n_samples_in_class, n_bands)

        # 转置为 (n_bands, n_samples_in_class) 并创建DataFrame
        df = pd.DataFrame(
            data=class_data.T,  # 转置后行是波段，列是样本
            columns=[f"1_{i + 1}" for i in range(class_data.shape[0])]  # 列名示例：1_1, 1_2...
        )
        df.insert(0, "Band", band_names)  # 插入波段名称列

        # 保存为Excel（第一行全为1）
        file_path = os.path.join(save_dir, f"{class_name}_processed.xlsx")
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 写入数据（从第二行开始）
            df.to_excel(writer, index=False, startrow=1)
            # 操作工作表，设置第一行全为1
            worksheet = writer.sheets['Sheet1']
            for col in range(1, df.shape[1]):
                worksheet.cell(row=1, column=col + 1, value=1)  # 第一行从第二列开始填1
            # 设置第一列标题
            worksheet.cell(row=1, column=1, value="Band")


    # ------------------
    # 转换为Tensor
    # ------------------
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)

    # ------------------
    # 动态增强配置
    # ------------------
    if dynamic_aug:
        aug = SpectralAugmentation(**(aug_params or {}))
        train_dataset = AugmentedDataset(train_x_tensor, train_y_tensor, augment=aug)
    else:
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)

    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

    # ------------------
    # 构建DataLoader
    # ------------------
    tr_batch_size = min(batch_size, len(train_x))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=tr_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    print(f"[预处理+增强后] 训练样本数: {len(train_dataset)}")
    print(f"[预处理后] 测试样本数: {len(test_dataset)}")

    return train_dataloader, test_dataloader

def load_hyper_save_aug(data_dir, class_names, tr_percent, batch_size,
               dynamic_aug=False, static_aug=False, augment_times=3,
               aug_params=None, preprocess_params=None):
    # 加载原始数据
    data, labels = loadData(data_dir, class_names)

    # 划分训练测试集（保持原始状态）
    if tr_percent < 1:
        train_x_raw, test_x_raw, train_y, test_y = train_test_split(
            data, labels, test_size=1 - tr_percent, stratify=labels, random_state=42)
    else:
        train_x_raw, test_x_raw, train_y, test_y = split_data_fix(data, labels, tr_percent)

    original_shape = train_x_raw.shape[1:]

    # 数据增强（在预处理之前）
    if static_aug and augment_times > 0:
        augmentor = SpectralAugmentation(**(aug_params or {}))
        # 初始化增强数据容器
        aug_train_x = []

        # 生成增强数据
        for _ in range(augment_times):
            # 对当前批次进行增强
            augmented_batch = augmentor.augment_batch(train_x_raw)
            aug_train_x.append(augmented_batch)

        # 合并原始数据和增强数据
        train_x_raw = np.concatenate([train_x_raw] + aug_train_x)
        train_y = np.concatenate([train_y] * (augment_times + 1))  # 标签同步扩展

        print(f"增强后训练集形状: {train_x_raw.shape}")

    # 预处理（在保存原始数据之后）
    if preprocess_params is not None:
        preprocessor = Preprocessor(**preprocess_params)
        train_x = preprocessor.fit_transform(train_x_raw)
        test_x = preprocessor.transform(test_x_raw)
    else:
        train_x = train_x_raw
        test_x = test_x_raw

    # 保存增强后的原始数据到Excel（预处理前）
    save_dir = os.path.join(data_dir, "augmented_data")
    os.makedirs(save_dir, exist_ok=True)

    # 获取波段信息（示例：假设是1050-1650nm的601个波段）
    wavelengths = np.linspace(1050, 1650, train_x_raw.shape[1])

    for class_idx, class_name in enumerate(class_names):
        # 获取该类别的数据索引
        mask = (train_y == class_idx)
        class_data = train_x_raw[mask]  # 使用原始数据，未经过预处理的

        # 创建DataFrame
        df = pd.DataFrame(class_data.T,  # 转置为（波段数, 样本数）
                          columns=[f"Sample_{i + 1}" for i in range(class_data.shape[0])])
        df.insert(0, "Wavelength(nm)", wavelengths)  # 添加波长列

        # 保存到Excel
        save_path = os.path.join(save_dir, f"{class_name}_augmented.xlsx")
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Spectra')
            print(f"保存增强数据到 {save_path}")

    # ------------------
    # 转换为Tensor
    # ------------------
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)

    # ------------------
    # 动态增强配置
    # ------------------
    if dynamic_aug:
        aug = SpectralAugmentation(**(aug_params or {}))
        train_dataset = AugmentedDataset(train_x_tensor, train_y_tensor, augment=aug)
    else:
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)

    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

    # ------------------
    # 构建DataLoader
    # ------------------
    tr_batch_size = min(batch_size, len(train_x))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=tr_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    print(f"[预处理+增强后] 训练样本数: {len(train_dataset)}")
    print(f"[预处理后] 测试样本数: {len(test_dataset)}")

    return train_dataloader, test_dataloader

def tsne(features, n_components=2, perplexity=20):
    """t-SNE降维工具函数"""
    return TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42
    ).fit_transform(features)

if __name__ == "__main__":
    AUG_PARAMS = {
        'noise_std': 0.02,
        'shift_max': 5,
        'warp_scale': 0.15,
        'mask_ratio': 0.1
    }

    cfg_msc_only = {
        'steps': ['MSC'],
        'MSC': True
    }

    data_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2/YuanYi"
    class_names = ["Healthy", "Asymptomatic", "Moderate_infected", "Severely_infected"]

    train_dataloader, test_dataloader = load_hyper(data_dir, class_names, 0.7, 32, dynamic_aug = False, static_aug = True, augment_times=3,  # 每个样本生成3个增强版本
        aug_params=AUG_PARAMS, preprocess_params=cfg_msc_only)

    # 检查 DataLoader
    print(f"训练集批次数: {len(train_dataloader)}")
    print(f"测试集批次数: {len(test_dataloader)}")

    # 验证样本数量
    print(f"[增强后] 训练样本总数: {len(train_dataloader.dataset)}")
    print(f"[原始] 测试样本数量: {len(test_dataloader.dataset)}")

    # 查看第一个批次数据
    sample_batch = next(iter(train_dataloader))
    print(f"\n批次数据维度: {sample_batch[0].shape}")
    print(f"批次标签分布: {np.bincount(sample_batch[1].numpy())}")


def integrate_bands(method_bands):
    """整合多方法选择的波段"""
    all_bands = []
    for method, bands in method_bands.items():
        all_bands.extend(bands)
    return np.unique(all_bands)


def auto_cluster_bands(bands, eps=10, min_samples=2):
    """自动聚类划分波段区域"""
    X = np.array(bands).reshape(-1, 1)

    # 动态调整eps参数
    eps = max(eps, int(np.median(np.diff(np.sort(bands)))))

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # 生成区域字典
    regions = defaultdict(list)
    for band, label in zip(bands, labels):
        if label != -1:  # 忽略噪声点
            regions[label].append(band)

    # 转换为区间表示
    sorted_regions = []
    for cluster in regions.values():
        cluster = sorted(cluster)
        sorted_regions.append((min(cluster), max(cluster)))

    # 按起始波长排序
    return sorted(sorted_regions, key=lambda x: x[0])

import torch.nn as nn
class AttributionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, cls_output = self.model(x)
        return cls_output


def calculate_region_importance(model, regions, X_sample, sorted_bands, device):
    """计算各区域重要性"""
    X_tensor = torch.tensor(X_sample, dtype=torch.float32)
    X_tensor = X_tensor.unsqueeze(1)  # [batch, 1, n_bands]
    X_tensor = X_tensor.to(device).requires_grad_(True)

    # 使用包装模型
    attribution_model = AttributionWrapper(model).to(device)
    attribution_model.eval()


    # 使用DeepLIFT计算特征重要性
    dl = DeepLift(attribution_model)
    attr = dl.attribute(X_tensor, target=0, return_convergence_delta=False)

    # 聚合区域重要性
    region_importances = []
    for (start, end) in regions:
        # 获取该区域在原始波段中的索引
        region_indices = [i for i, band in enumerate(sorted_bands) if start <= band <= end]
        if not region_indices:
            region_importances.append(0.0)
            continue
        # 计算区域平均重要性
        region_importance = torch.mean(torch.abs(attr[:, region_indices])).item()
        region_importances.append(region_importance)

    # 归一化
    total = sum(region_importances)
    return [imp / total for imp in region_importances]


def dynamic_band_selection(regions, importances, total_bands=4, strategy='proportional'):
    """动态波段选择核心逻辑"""
    selected = []

    if strategy == 'proportional':
        # 计算每个区域应选数量
        probs = np.array(importances) / sum(importances)
        num_to_select = np.round(probs * total_bands).astype(int)

        # 处理四舍五入导致的数量不足
        while sum(num_to_select) < total_bands:
            max_idx = np.argmax(probs - num_to_select / total_bands)
            num_to_select[max_idx] += 1

        # 从每个区域随机选择
        for i, (start, end) in enumerate(regions):
            candidates = list(range(start, end + 1))
            if len(candidates) == 0:
                continue
            selected += np.random.choice(candidates,
                                         size=num_to_select[i],
                                         replace=False).tolist()

    elif strategy == 'topk':
        # 选择重要性最高的前k个区域
        sorted_regions = sorted(zip(regions, importances), key=lambda x: -x[1])
        for (start, end), _ in sorted_regions[:total_bands]:
            selected.append((start + end) // 2)  # 取区域中心

    return sorted(selected[:total_bands])

def extract_data_from_dataloader(dataloader):
    """从DataLoader中提取numpy数据"""
    features = []
    labels = []
    for batch in dataloader:
        x, y = batch
        features.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)
