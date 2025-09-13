import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from openpyxl import Workbook

from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


from matplotlib.font_manager import FontProperties



def detect_outliers(samples, contamination=0.05):
    """使用孤立森林检测异常样本"""
    clf = IsolationForest(n_estimators=100,
                          contamination=contamination,
                          random_state=42)
    outlier_mask = clf.fit_predict(samples) == -1
    return samples[~outlier_mask], outlier_mask


def save_cleaned_data(original_df, cleaned_samples, output_path):
    """
    保存清洗后的数据，保持原始格式
    Args:
        original_df: 原始DataFrame（带波段索引）
        cleaned_samples: 清洗后的样本数据 (n_samples_clean, n_bands)
        output_path: 输出文件路径
    """
    # 转置回原始格式 (波段, 样本)
    cleaned_data = cleaned_samples.T  # (n_bands, n_samples_clean)

    # 重建DataFrame
    cleaned_df = pd.DataFrame(
        cleaned_data,
        index=original_df.index,  # 保持原始波段索引
        columns=[f"1_{i + 1}" for i in range(cleaned_data.shape[1])]  # 生成唯一列名
    )

    # 保存数据（使用openpyxl处理格式）
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        cleaned_df.to_excel(writer, index=True, header=True)

        # 修改标题行为全1显示
        ws = writer.sheets['Sheet1']
        for col in range(2, cleaned_df.shape[1] + 2):  # 从B列开始
            ws.cell(row=1, column=col, value=1)


# def visualize_outliers(original_samples, cleaned_samples, class_name, output_dir):
    # """PCA可视化异常值剔除效果"""
    # # 设置中文字体（多平台兼容方案）
    # try:
    #     # Windows系统首选字体
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
    #     plt.rcParams['axes.unicode_minus'] = False
    # except:
    #     try:
    #         # macOS系统字体
    #         plt.rcParams['font.family'] = 'Arial Unicode MS'
    #     except:
    #         # Linux系统字体（需要安装）
    #         plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    #
    # # 合并数据用于统一PCA模型
    # combined = np.vstack([original_samples, cleaned_samples])
    # pca = PCA(n_components=2).fit(combined)
    #
    # # 转换数据
    # original_pca = pca.transform(original_samples)
    # cleaned_pca = pca.transform(cleaned_samples)
    #
    # # 生成异常掩码
    # outlier_mask = ~np.isin(original_samples, cleaned_samples).all(axis=1)
    #
    # # 创建可视化
    # plt.figure(figsize=(10, 6))
    # plt.scatter(original_pca[:, 0], original_pca[:, 1],
    #             c=outlier_mask, cmap='viridis', alpha=0.6)
    # plt.title(f'{class_name} - 异常值分布 (红色为剔除样本)')
    # plt.colorbar(label='是否异常 (1=异常)')
    # plt.savefig(os.path.join(output_dir, f"{class_name}_outliers.png"))
    # plt.close()


def configure_chinese_font():
    """专业级中文字体配置方案"""
    try:
        # 方案1：使用绝对路径指定字体文件
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows系统黑体
        if os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return True

        # 方案2：Linux/macOS备用方案
        font_path = '/System/Library/Fonts/STHeiti Medium.ttc'  # macOS系统字体
        if os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            return True

        # 方案3：使用思源黑体（需预先下载）
        font_path = 'SourceHanSansSC-Regular.otf'  # 需要提前放置字体文件
        if os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            return True

    except Exception as e:
        print(f"字体配置失败: {e}")

    # 最终回退方案：尝试系统默认中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    return False


def save_figure(fig, path, dpi=100):
    """安全保存图片（解决PDF字体嵌入问题）"""
    plt.rcParams['pdf.fonttype'] = 42  # 确保嵌入TrueType字体
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    fig.savefig(path,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.1,
                metadata={
                    'CreationDate': None,
                    'Producer': 'Your Research Tool',
                    'Author': 'Your Name',
                    'Title': 'Spectral Analysis'
                })

def visualize_outliers(original_samples, cleaned_samples, class_name, output_dir):
    """学术论文级可视化函数"""
    # # 设置中 文字体（多平台兼容方案）
    if not configure_chinese_font():
        print("警告：中文显示可能不正常，请安装中文字体")


    # ==================== 高级配置 ====================
    plt.style.use('seaborn-paper')  # 使用学术论文专用样式
    COLOR_PALETTE = ['#2F4F4F', '#CD5C5C']  # 学术色板（铁灰/砖红）
    FIG_SIZE = (8, 5)  # Nature期刊推荐比例
    DPI = 1200  # 出版级分辨率
    FONT_CONFIG = {
        'family': 'Arial',  # 国际期刊通用字体
        'size': 10,
        'weight': 'normal'
    }

    # ==================== 字体设置 ====================
    plt.rc('font', **FONT_CONFIG)
    plt.rc('axes', titlesize=12, labelsize=10, linewidth=0.8)
    plt.rc('lines', linewidth=1.2, markersize=6)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('legend', fontsize=9, frameon=True, framealpha=0.8)

    # ==================== 数据准备 ====================
    combined = np.vstack([original_samples, cleaned_samples])
    pca = PCA(n_components=2).fit(combined)
    original_pca = pca.transform(original_samples)
    cleaned_pca = pca.transform(cleaned_samples)
    outlier_mask = ~np.isin(original_samples, cleaned_samples).all(axis=1)
    variance_ratio = pca.explained_variance_ratio_

    # ==================== 绘图区域 ====================
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=100)  # 初始dpi用于计算

    # 主散点图（带边缘直方图）
    main_scatter = ax.scatter(
        original_pca[:, 0], original_pca[:, 1],
        c=outlier_mask, cmap=ListedColormap(COLOR_PALETTE),
        alpha=0.8, edgecolors='w', linewidths=0.5,
        zorder=2
    )

    # 添加密度等高线
    # kde = gaussian_kde(cleaned_pca.T)
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    # xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([xx.ravel(), yy.ravel()])
    # f = np.reshape(kde(positions).T, xx.shape)
    # ax.contour(xx, yy, f, levels=4, colors=COLOR_PALETTE[0],
    #            linewidths=0.8, alpha=0.6, zorder=1)

    # ==================== 样式优化 ====================
    # 坐标轴设置
    ax.set_xlabel(f"PC1 ({variance_ratio[0] * 100:.1f}%)", labelpad=3)
    ax.set_ylabel(f"PC2 ({variance_ratio[1] * 100:.1f}%)", labelpad=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True, pad=2)

    # 网格线
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal samole',
               markerfacecolor=COLOR_PALETTE[0], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Abnormal  sample',
               markerfacecolor=COLOR_PALETTE[1], markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              facecolor='white', framealpha=0.9)

    # 颜色条
    cbar = plt.colorbar(main_scatter, ax=ax, pad=0.02)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Normal', 'Abnormal'])
    cbar.outline.set_linewidth(0.5)

    # ==================== 注释与统计 ====================
    stats_text = f"""
    Total samples: {len(original_samples)}
    Outliers removed: {sum(outlier_mask)} ({sum(outlier_mask) / len(original_samples):.1%})
    Retained samples: {len(cleaned_samples)}
    """
    ax.text(0.98, 0.15, stats_text.strip(), transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8,
                      edgecolor='gray', boxstyle='round,pad=0.3'))

    # ==================== 输出设置 ====================
    plt.tight_layout(pad=1.5)

    # 保存多种格式
    base_path = os.path.join(output_dir, f"{class_name}_outliers")

    plt.savefig(f"{base_path}.png", dpi=DPI, bbox_inches='tight')
    # save_figure(fig, f"{base_path}.png")
    # plt.savefig(f"{base_path}.pdf", dpi=DPI, bbox_inches='tight')  # 矢量图
    # plt.savefig(f"{base_path}.tiff", dpi=DPI, bbox_inches='tight',  # 位图
    #             pil_kwargs={"compression": "tiff_lzw"})
    plt.close()

def process_and_save_data(input_dir, output_dir, class_names):
    """主处理流程"""
    os.makedirs(output_dir, exist_ok=True)

    for class_name in class_names:
        # 加载原始数据
        input_path = os.path.join(input_dir, f"{class_name}.xlsx")
        df = pd.read_excel(input_path, index_col=0, header=0)
        original_samples = df.T.values  # (n_samples, n_bands)

        # 异常值检测
        cleaned_samples, outlier_mask = detect_outliers(original_samples)
        print(f"{class_name}: 原始样本 {len(original_samples)} → 保留 {len(cleaned_samples)}")

        # 保存清洗数据
        output_path = os.path.join(output_dir, f"{class_name}.xlsx")
        save_cleaned_data(df, cleaned_samples, output_path)

        # 生成可视化
        visualize_outliers(original_samples, cleaned_samples, class_name, output_dir)


# 使用示例
if __name__ == "__main__":
    # 1000-1700nm
    # input_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2/YuanYi"  # 原始数据目录
    # output_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2/YuanYi/cleaned_data"  # 清洗后数据目录
    # class_names = ["Healthy", "Asymptomatic", "Moderate_infected", "Severely_infected"]

    # 1D
    # input_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\YuanYi\pro_data\processed_data\3.1D"  # 原始数据目录
    # output_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\YuanYi\pro_data\processed_data\3.1D\clean_data"  # 清洗后数据目录
    # class_names = ["Healthy", "Asymptomatic", "Moderate_infected", "Severely_infected"]

    # 无症状细分
    # input_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2\YuanYi/2.Asynptomatic/1.raw_data"  # 原始数据目录
    # output_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2\YuanYi/2.Asynptomatic/2.cleaned_data"  # 清洗后数据目录
    # class_names = ["AE", "AM", "AL"]

    # 400-900nm
    # input_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2/YuanYi/1.400-900nm/0.raw_data"  # 原始数据目录
    # output_dir = "E:/0_Exp/2.data/0.banana_foc_202412/1.Task_2/YuanYi/1.400-900nm/cleaned_data"  # 清洗后数据目录
    # class_names = ["Healthy", "Asymptomatic", "Infected"]

    # 7_class
    # input_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\7_CLASS\2.extract_raw_data"  # 原始数据目录
    # output_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\7_CLASS\3.cleaned_data"  # 清洗后数据目录
    # class_names = ["H", "AE", "AM", "AL", "MI", "MO", "SE"]

    # 5_class
    # input_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\5_CLASS\2.extract_raw_data"  # 原始数据目录
    # output_dir = r"E:\0_Exp\2.data\0.banana_foc_202412\1.Task_2\202505_New_Data_EXP\5_CLASS\3.cleaned_data"  # 清洗后数据目录
    # class_names = ["H", "A", "MI", "MO", "SE"]

    # 400-900nm
    input_dir = r"E:\0_Exp\3.Modeling_experiments\1.400-900nm_EXP\0.Dataset_raw"  # 原始数据目录
    output_dir = r"E:\0_Exp\3.Modeling_experiments\1.400-900nm_EXP\2.clean_data"  # 清洗后数据目录
    class_names = ["H", "A", "MI", "MO", "SE"]


    process_and_save_data(input_dir, output_dir, class_names)




