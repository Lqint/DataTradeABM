import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

# 设置中文字体（根据系统环境调整，Windows常用SimHei，Mac常用Heiti TC）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
# 设置学术配色
sns.set_theme(style="whitegrid", font="SimHei")
PALETTE = sns.color_palette("deep")

def load_data(output_dir="experiment_output"):
    # 1. 加载 Summary 数据
    summary_path = os.path.join(output_dir, "summary.csv")
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
    else:
        df_summary = None

    # 2. 加载 History 数据
    history_files = glob.glob(os.path.join(output_dir, "*_history.csv"))
    history_list = []
    
    print(f"发现 {len(history_files)} 个历史文件，开始加载...")
    for f in history_files:
        try:
            basename = os.path.basename(f)
            parts = basename.rsplit('_', 2) 
            if len(parts) < 3: continue
            
            group_name = parts[0]
            run_id = int(parts[1])
            
            cols = ['step', 'trade_volume', 'trust', 'providers', 'third_party_share', 'maturity', 'entry_rate', 'exit_rate']
            df = pd.read_csv(f, usecols=cols)
            df['group'] = group_name
            df['run_id'] = run_id
            history_list.append(df)
        except Exception as e:
            continue
            
    if history_list:
        df_history = pd.concat(history_list, ignore_index=True)
    else:
        df_history = None
        
    return df_summary, df_history

def load_sweep_data(output_dir="experiment_output"):
    sweep_path = os.path.join(output_dir, "parameter_sweep.csv")
    if os.path.exists(sweep_path):
        print(f"加载参数扫描数据: {sweep_path}")
        return pd.read_csv(sweep_path)
    else:
        print("未找到 parameter_sweep.csv，无法绘制倒U型曲线。")
        return None

def plot_proposition_1_marginal_effect(df_history, output_dir):
    """命题一：固定自营比例，自营边际效应随成熟度演化"""
    if df_history is None: return
    
    # 对比组：G1 (0.5) vs C1 (0.1)
    # 使用 pivot_table 对齐步数并计算均值
    df_pivot = df_history.pivot_table(index='step', columns='group', values=['trade_volume', 'maturity'], aggfunc='mean')
    
    # 提取需要的列
    if ('trade_volume', 'G1_initial_high_self') not in df_pivot.columns or \
       ('trade_volume', 'C1_low_self') not in df_pivot.columns:
        print("警告：缺少 G1 或 C1 组数据，无法绘制命题一图表")
        return

    vol_g1 = df_pivot[('trade_volume', 'G1_initial_high_self')]
    vol_c1 = df_pivot[('trade_volume', 'C1_low_self')]
    maturity = df_pivot[('maturity', 'G1_initial_high_self')] # 使用 G1 的成熟度作为 X 轴
    
    # 计算边际效应 Delta = G1 - C1
    delta_volume = vol_g1 - vol_c1
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 绘制曲线
    ax.plot(maturity, delta_volume, color=PALETTE[3], linewidth=2.5, label='净边际效应 (G1 - C1)')
    
    # 添加零线
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # 标注正负区域
    ax.fill_between(maturity, 0, delta_volume, where=(delta_volume > 0), 
                    color=PALETTE[2], alpha=0.2, label='正向促进区')
    ax.fill_between(maturity, 0, delta_volume, where=(delta_volume <= 0), 
                    color=PALETTE[0], alpha=0.2, label='负向挤出区')
    
    ax.set_title('图A (命题一). 自营边际效应随平台成熟度的演化', fontsize=14)
    ax.set_xlabel('平台成熟度 (Maturity)', fontsize=12)
    ax.set_ylabel('交易规模净差异 (Delta Volume)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 标注翻转点 (过滤掉成熟度小于0.1的早期震荡点)
    cross_idx = np.where(np.diff(np.sign(delta_volume)))[0]
    
    # 找到第一个成熟度 > 0.1 的翻转点
    target_idx = -1
    for idx in cross_idx:
        if maturity.iloc[idx] > 0.1:
            target_idx = idx
            break
            
    # 如果没找到 > 0.1 的，但有翻转点，就取最后一个
    if target_idx == -1 and len(cross_idx) > 0:
        target_idx = cross_idx[-1]
        
    if target_idx != -1:
        threshold_m = maturity.iloc[target_idx]
        ax.annotate(f'效应翻转点 M ≈ {threshold_m:.2f}', 
                    xy=(threshold_m, 0), xytext=(threshold_m+0.1, 5),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FigA_Proposition1_Marginal_Evolution.png"), bbox_inches='tight')
    print("生成图A (命题一) 完成")

def plot_proposition_2_inverted_u(df_sweep, output_dir):
    """命题二：固定成熟度，自营比例对交易规模的边际效应"""
    if df_sweep is None: return
    
    # 定义三个成熟度切片 - 恢复为全景视角以检查新参数下的整体变化
    target_maturities = [0.3, 0.5, 0.8]
    labels = ['低成熟度 (M≈0.3)', '中成熟度 (M≈0.5)', '高成熟度 (M≈0.8)']
    colors = [PALETTE[2], PALETTE[1], PALETTE[0]] # 绿 -> 橙 -> 蓝
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # 添加零线
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    
    for idx, target_m in enumerate(target_maturities):
        # 筛选接近目标成熟度的数据 (±0.05)
        mask = (df_sweep['maturity'] >= target_m - 0.05) & (df_sweep['maturity'] <= target_m + 0.05)
        df_slice = df_sweep[mask]
        
        # 按自营比例分组计算均值
        df_agg = df_slice.groupby('self_share')['trade_volume'].mean().reset_index()
        
        if len(df_agg) < 3: continue # 数据点太少无法绘图
        
        # 平滑曲线处理 (B-Spline)
        x = df_agg['self_share']
        y = df_agg['trade_volume']
        # Increase resolution for smoother zoomed-in plot
        x_smooth = np.linspace(x.min(), x.max(), 1000)
        y_marginal = None
        
        try:
            spl = make_interp_spline(x, y, k=3)
            # 计算导数 (边际效应)
            spl_deriv = spl.derivative()
            y_marginal = spl_deriv(x_smooth)
        except Exception:
            # Fallback: 使用梯度计算导数
            dy = np.gradient(y, x)
            y_marginal = np.interp(x_smooth, x, dy)
        
        # 绘制曲线
        ax.plot(x_smooth, y_marginal, color=colors[idx], linewidth=2.5, label=labels[idx])
        
        # 寻找并标注最高的极值点 (只标注内部极值点，不标注边界)
        peaks, _ = find_peaks(y_marginal)
        
        if len(peaks) > 0:
            # 在所有极值点中找到最高的那个
            best_peak_idx = peaks[np.argmax(y_marginal[peaks])]
            
            peak_x = x_smooth[best_peak_idx]
            peak_y = y_marginal[best_peak_idx]
            
            # 标记点
            ax.plot(peak_x, peak_y, 'o', color=colors[idx], markersize=8)
            # 垂线
            ax.vlines(peak_x, np.min(y_marginal), peak_y, color=colors[idx], linestyle=':', alpha=0.5)
            # 文本标注
            offset = (np.max(y_marginal) - np.min(y_marginal))*0.05
            ax.annotate(f'ω*={peak_x:.2f}', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + offset),
                        ha='center', fontsize=9, color=colors[idx], fontweight='bold')

    ax.set_title('图B (命题二). 不同成熟度下自营比例对交易规模的边际效应', fontsize=14)
    ax.set_xlabel('自营展示比例 (Self-Share)', fontsize=12)
    ax.set_ylabel('交易规模边际效应 (Marginal Effect)', fontsize=12)
    # 暂时移除 X 轴限制，以便全景观察新逻辑的效果
    # ax.set_xlim(0.6, 0.7)
    
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FigB_Proposition2_Inverted_U.png"), bbox_inches='tight')
    print("生成图B (命题二) 完成")

if __name__ == "__main__":
    OUTPUT_DIR = "experiment_output"
    FINAL_FIGURES_DIR = "final_figures"
    
    # 确保输出目录存在
    if not os.path.exists(FINAL_FIGURES_DIR):
        os.makedirs(FINAL_FIGURES_DIR)
    
    # 1. 加载常规实验数据 (用于图A)
    _, df_history = load_data(OUTPUT_DIR)
    if df_history is not None:
        plot_proposition_1_marginal_effect(df_history, FINAL_FIGURES_DIR)
        
    # 2. 加载参数扫描数据 (用于图B)
    df_sweep = load_sweep_data(OUTPUT_DIR)
    if df_sweep is not None:
        plot_proposition_2_inverted_u(df_sweep, FINAL_FIGURES_DIR)
