import pandas as pd
import glob
import os
from datetime import datetime

def get_timestamp_from_filename(filename):
    try:
        # 检查文件名是否符合预期格式
        if not filename.startswith(('record_', 'pixels_')) or not filename.endswith('.csv'):
            return None
        # 从文件名中提取时间戳 (格式: YYYYMMDD_HHMMSS)
        timestamp_str = filename.split('_')[-2] + '_' + filename.split('_')[-1].split('.')[0]
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except (IndexError, ValueError):
        return None

# 获取所有record文件（包括两个子目录）
record_files = []
# record_files.extend(glob.glob('records625/*.csv'))
record_files.extend(glob.glob('records626/*.csv'))

# 过滤掉不符合命名规则的文件
record_files = [f for f in record_files if os.path.basename(f).startswith('record_')]

print("找到的record文件：")
for f in record_files:
    print(f"- {os.path.basename(f)}")

# 请输入pixel文件所在的目录
pixel_dir = input("\n请输入pixel文件所在的目录路径：")
all_pixel_files = glob.glob(os.path.join(pixel_dir, '*.csv'))

# 过滤掉不符合命名规则的文件
pixel_files = [f for f in all_pixel_files if os.path.basename(f).startswith('pixels_')]

print("\n找到的pixel文件：")
for f in pixel_files:
    print(f"- {os.path.basename(f)}")

# 创建文件名到时间戳的映射
record_timestamps = {}
for f in record_files:
    timestamp = get_timestamp_from_filename(os.path.basename(f))
    if timestamp:
        record_timestamps[timestamp] = f

pixel_timestamps = {}
for f in pixel_files:
    timestamp = get_timestamp_from_filename(os.path.basename(f))
    if timestamp:
        pixel_timestamps[timestamp] = f

# 匹配文件并添加rpm
for pixel_time, pixel_file in pixel_timestamps.items():
    # 找到最接近的record文件
    time_diffs = [(abs((record_time - pixel_time).total_seconds()), record_file) 
                  for record_time, record_file in record_timestamps.items()]
    closest_record = min(time_diffs, key=lambda x: x[0])[1]
    
    # 如果时间差在60秒以内，认为是匹配的文件
    if min(time_diffs, key=lambda x: x[0])[0] <= 60:
        print(f"\n处理文件对：")
        print(f"Pixel文件: {os.path.basename(pixel_file)}")
        print(f"Record文件: {os.path.basename(closest_record)}")
        
        # 读取record文件获取rpm
        record_df = pd.read_csv(closest_record)
        if record_df.empty or 'rpm' not in record_df.columns or record_df['rpm'].dropna().empty:
            print("  警告：Record文件中没有有效的rpm数据，已跳过该文件对。")
            continue
        rpm = record_df['rpm'].dropna().iloc[0]
        
        # 读取pixel文件
        pixel_df = pd.read_csv(pixel_file)
        
        # 添加rpm列
        pixel_df['rpm'] = rpm
        
        # 保存更新后的pixel文件
        pixel_df.to_csv(pixel_file, index=False)
        print(f"已添加RPM: {rpm}")
    else:
        print(f"\n警告：找不到匹配的record文件：{os.path.basename(pixel_file)}")
        print(f"最小时间差: {min(time_diffs, key=lambda x: x[0])[0]:.2f} 秒") 