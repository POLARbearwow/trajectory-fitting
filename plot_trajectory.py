import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# 读取所有CSV文件
csv_files = glob.glob('records/*.csv')

# 获取RPM为3636的文件
target_rpm = 3636
rpm_files = []
for file in csv_files:
    df = pd.read_csv(file)
    if df['rpm'].iloc[0] == target_rpm:
        rpm_files.append(file)

# 创建2行3列的子图（上面一行显示轨迹，下面一行显示时间间隔）
plt.ion()  # 打开交互模式
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Trajectory Plot and Time Intervals (RPM: {target_rpm})', fontsize=16)

# 为不同文件创建不同的颜色
colors = plt.cm.rainbow(np.linspace(0, 1, len(rpm_files)))

# 在对应的子图中绘制数据
for file_idx, file in enumerate(rpm_files):
    df = pd.read_csv(file)
    
    # 计算时间间隔（毫秒）
    df['time_diff'] = df['timestamp_ms'].diff()
    
    # 上面的子图绘制轨迹
    axes[0, file_idx].scatter(-df['new_y'], df['new_x'],
                             color=colors[file_idx],
                             s=20,
                             alpha=0.7)
    
    # 设置轨迹图的标题和标签
    axes[0, file_idx].set_title(f'File {file_idx+1}')
    axes[0, file_idx].set_xlabel('New Y Position (Negated)')
    axes[0, file_idx].set_ylabel('New X Position')
    axes[0, file_idx].grid(True, linestyle='--', alpha=0.7)
    
    # 下面的子图绘制时间间隔
    axes[1, file_idx].plot(range(len(df['time_diff'])), df['time_diff'], 'b.')
    axes[1, file_idx].set_title(f'Time Intervals')
    axes[1, file_idx].set_xlabel('Sample Index')
    axes[1, file_idx].set_ylabel('Time Interval (ms)')
    axes[1, file_idx].grid(True, linestyle='--', alpha=0.7)
    
    # 打印时间间隔的统计信息
    print(f"\nFile {file_idx+1} time intervals statistics:")
    print(f"Mean interval: {df['time_diff'].mean():.2f} ms")
    print(f"Min interval: {df['time_diff'].min():.2f} ms")
    print(f"Max interval: {df['time_diff'].max():.2f} ms")
    print(f"Std interval: {df['time_diff'].std():.2f} ms")

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()

# 保持窗口打开
input("Press Enter to close the plot...") 