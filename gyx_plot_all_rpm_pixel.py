import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import to_rgba
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ----------------- 配置 -----------------
# 像素文件所在目录(含 pixels_*.csv)
BASE_DIR = '/Users/niuniu/Desktop/trajectory fitting/records626'
# 每种RPM使用的色图
COLOR_MAP = plt.cm.tab20  # 最多支持 20 种颜色
POINT_SIZE = 8
LINE_WIDTH = 1.0
# 折线透明度
POLYLINE_ALPHA = 0.3
# 线宽设置
CURVE_WIDTH = 2.0          # 拟合曲线线宽
POLYLINE_WIDTH = 3.0       # 折线线宽

def lighten_color(color, amount: float = 0.5):
    r, g, b, a = to_rgba(color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b, a)

def get_timestamp_from_filename(filename: str):
    """解析文件名中的时间戳，pixels_YYYYMMDD_HHMMSS.csv"""
    try:
        if not filename.startswith('pixels_'):
            return None
        date_part, time_part_with_ext = filename.split('_')[-2], filename.split('_')[-1]
        time_part = time_part_with_ext.split('.')[0]
        dt_str = f"{date_part}_{time_part}"
        return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    except Exception:
        return None

def load_pixel_files(base_dir):
    pixels = []
    for fpath in glob.glob(os.path.join(base_dir, 'pixels_*.csv')):
        fname = os.path.basename(fpath)
        ts = get_timestamp_from_filename(fname)
        if ts is None:
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f'读取 {fpath} 失败: {e}')
            continue
        # 过滤无效点
        df = df[(df['ball_u'] != -999) & (df['ball_v'] != -999)]
        if df.empty:
            continue
        df['timestamp'] = df['timestamp_ms'] if 'timestamp_ms' in df.columns else np.arange(len(df))
        df['file_ts'] = ts
        df['source_file'] = fname
        pixels.append(df)
    if not pixels:
        return pd.DataFrame()
    return pd.concat(pixels, ignore_index=True)

def extract_trajectory_parameters(x_vals, y_vals):
    """从轨迹数据中提取二次函数参数"""
    try:
        # 拟合二次函数 y = ax^2 + bx + c
        coeffs = np.polyfit(x_vals, y_vals, deg=2)
        a, b, c = coeffs
        
        # 计算物理意义的参数
        # 顶点坐标
        vertex_x = -b / (2 * a) if a != 0 else np.mean(x_vals)
        vertex_y = a * vertex_x**2 + b * vertex_x + c
        
        # 初始位置 (假设x=0时)
        if len(x_vals) > 0:
            initial_x = x_vals[0]
            initial_y = y_vals[0]
        else:
            initial_x, initial_y = 0, c
            
        # 计算R²
        y_pred = np.polyval(coeffs, x_vals)
        r2 = r2_score(y_vals, y_pred) if len(y_vals) > 1 else 0
        
        return {
            'a': a, 'b': b, 'c': c,
            'vertex_x': vertex_x, 'vertex_y': vertex_y,
            'initial_x': initial_x, 'initial_y': initial_y,
            'r2': r2,
            'coeffs': coeffs
        }
    except:
        return None

def physics_based_model(rpm, A, B, C, D):
    """基于物理的模型：假设初始速度与RPM成正比"""
    # v0 = A * rpm + B (初始速度)
    # 对于抛物运动，系数a与初始速度的平方成反比
    v0 = A * rpm + B
    a = -C / (v0**2 + 1e-6)  # 避免除零
    return a + D  # D是修正项

def polynomial_model_2nd(rpm, a0, a1, a2):
    """二次多项式模型"""
    return a0 + a1 * rpm + a2 * rpm**2

def polynomial_model_3rd(rpm, a0, a1, a2, a3):
    """三次多项式模型"""
    return a0 + a1 * rpm + a2 * rpm**2 + a3 * rpm**3

def exponential_model(rpm, A, B, C):
    """指数模型"""
    return A + B * np.exp(C * rpm)

def power_model(rpm, A, B, C):
    """幂函数模型"""
    return A + B * (rpm + 1)**C  # +1避免rpm=0时的问题

def fit_rpm_relationship(rpm_data, param_data, param_name):
    """拟合RPM与轨迹参数的关系"""
    models = {
        'Linear': lambda x, a, b: a * x + b,
        'Quadratic': polynomial_model_2nd,
        'Cubic': polynomial_model_3rd,
        'Physics': physics_based_model,
        'Exponential': exponential_model,
        'Power': power_model
    }
    
    results = {}
    
    for model_name, model_func in models.items():
        try:
            # 设置初始参数猜测
            if model_name == 'Linear':
                p0 = [1, 0]
            elif model_name == 'Quadratic':
                p0 = [0, 0, 0]
            elif model_name == 'Cubic':
                p0 = [0, 0, 0, 0]
            elif model_name == 'Physics':
                p0 = [0.01, 100, 1000, 0]
            elif model_name == 'Exponential':
                p0 = [np.mean(param_data), 1, 0.001]
            elif model_name == 'Power':
                p0 = [np.mean(param_data), 1, 1]
            
            popt, pcov = curve_fit(model_func, rpm_data, param_data, p0=p0, maxfev=5000)
            
            # 预测和评估
            y_pred = model_func(rpm_data, *popt)
            r2 = r2_score(param_data, y_pred)
            mse = mean_squared_error(param_data, y_pred)
            
            results[model_name] = {
                'params': popt,
                'covariance': pcov,
                'r2': r2,
                'mse': mse,
                'model_func': model_func
            }
            
        except Exception as e:
            print(f"模型 {model_name} 拟合参数 {param_name} 失败: {e}")
            continue
    
    # 选择最佳模型（基于R²）
    if results:
        best_model = max(results.keys(), key=lambda k: results[k]['r2'])
        return results, best_model
    else:
        return {}, None

def predict_trajectory_coefficients(rpm, fitted_models):
    """根据RPM预测轨迹系数"""
    coeffs = []
    param_names = ['a', 'b', 'c']
    
    for param in param_names:
        if param in fitted_models:
            best_model_name = fitted_models[param]['best_model']
            model_info = fitted_models[param]['models'][best_model_name]
            model_func = model_info['model_func']
            params = model_info['params']
            predicted_value = model_func(rpm, *params)
            coeffs.append(predicted_value)
        else:
            coeffs.append(0)  # 默认值
    
    return np.array(coeffs)

def main():
    if not os.path.isdir(BASE_DIR):
        print(f'目录不存在: {BASE_DIR}')
        return
    
    all_pixels = load_pixel_files(BASE_DIR)
    if all_pixels.empty:
        print('未找到有效的 pixel 数据')
        return

    rpm_values = sorted(all_pixels['rpm'].unique())
    colors = COLOR_MAP(np.linspace(0, 1, len(rpm_values)))
    rpm_color = dict(zip(rpm_values, colors))

    # 提取每个RPM下的轨迹参数
    rpm_params = []
    
    print("提取轨迹参数...")
    for rpm in rpm_values:
        df_rpm_all = all_pixels[all_pixels['rpm'] == rpm]
        file_groups = [(fname, grp.copy()) for fname, grp in df_rpm_all.groupby('source_file')]
        
        rpm_trajectories = []
        
        for fname, df_file in file_groups:
            if len(df_file) < 20:
                continue
                
            df_file.sort_values('timestamp', inplace=True)
            img_h = df_file['ball_v'].max() + 10
            x_vals = df_file['ball_u'].values
            y_vals = img_h - df_file['ball_v'].values
            
            if len(y_vals) == 0:
                continue
                
            idx_min = int(np.argmin(y_vals))
            x_plot = x_vals[:idx_min + 1]
            y_plot = y_vals[:idx_min + 1]
            
            if len(x_plot) < 20:
                continue
                
            params = extract_trajectory_parameters(x_plot, y_plot)
            if params and params['r2'] > 0.7:  # 只保留拟合度较好的轨迹
                params['rpm'] = rpm
                params['filename'] = fname
                rpm_trajectories.append(params)
        
        if rpm_trajectories:
            # 对同一RPM下的多条轨迹取平均
            avg_params = {
                'rpm': rpm,
                'a': np.mean([t['a'] for t in rpm_trajectories]),
                'b': np.mean([t['b'] for t in rpm_trajectories]),
                'c': np.mean([t['c'] for t in rpm_trajectories]),
                'vertex_x': np.mean([t['vertex_x'] for t in rpm_trajectories]),
                'vertex_y': np.mean([t['vertex_y'] for t in rpm_trajectories]),
                'count': len(rpm_trajectories)
            }
            rpm_params.append(avg_params)
            print(f"RPM {rpm}: {len(rpm_trajectories)} 条有效轨迹")

    if not rpm_params:
        print("未找到足够的有效轨迹数据")
        return

    # 转换为数组进行拟合
    rpm_array = np.array([p['rpm'] for p in rpm_params])
    a_array = np.array([p['a'] for p in rpm_params])
    b_array = np.array([p['b'] for p in rpm_params])
    c_array = np.array([p['c'] for p in rpm_params])

    print(f"\n拟合 {len(rpm_params)} 个RPM数据点...")
    
    # 拟合每个参数与RPM的关系
    fitted_models = {}
    
    for param_name, param_array in [('a', a_array), ('b', b_array), ('c', c_array)]:
        print(f"\n拟合参数 '{param_name}' 与 RPM 的关系:")
        models, best_model = fit_rpm_relationship(rpm_array, param_array, param_name)
        
        if best_model:
            fitted_models[param_name] = {
                'models': models,
                'best_model': best_model
            }
            
            print(f"  最佳模型: {best_model}")
            print(f"  R² = {models[best_model]['r2']:.4f}")
            print(f"  MSE = {models[best_model]['mse']:.6f}")
            print(f"  参数: {models[best_model]['params']}")

    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 记录原始数据的坐标范围，用于统一比例尺
    all_x_coords = []
    all_y_coords = []

    # 子图1: 原始轨迹
    for rpm in rpm_values:
        df_rpm_all = all_pixels[all_pixels['rpm'] == rpm]
        file_groups = [(fname, grp.copy()) for fname, grp in df_rpm_all.groupby('source_file')]
        file_groups_use = file_groups
        first_curve_plotted = False

        for fname, df_file in file_groups_use:
            if len(df_file) < 20:
                continue

            df_file.sort_values('timestamp', inplace=True)
            img_h = df_file['ball_v'].max() + 10
            x_vals = df_file['ball_u'].values
            y_vals = img_h - df_file['ball_v'].values

            if len(y_vals) == 0:
                continue

            idx_min = int(np.argmin(y_vals))
            x_plot = x_vals[:idx_min + 1]
            y_plot = y_vals[:idx_min + 1]

            if len(x_plot) < 20:
                continue

            # 收集坐标范围信息
            all_x_coords.extend(x_plot)
            all_y_coords.extend(y_plot)

            light_col = lighten_color(rpm_color[rpm], 0.5)
            ax1.plot(x_plot, y_plot, color=light_col, linewidth=POLYLINE_WIDTH, alpha=POLYLINE_ALPHA)

            coeffs = np.polyfit(x_plot, y_plot, deg=2)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min(x_plot), max(x_plot), 200)
            y_fit = poly(x_fit)

            label = f'RPM {rpm}' if not first_curve_plotted else None
            ax1.plot(x_fit, y_fit, color=rpm_color[rpm], linewidth=CURVE_WIDTH, label=label)
            first_curve_plotted = True

    ax1.set_title('Original Trajectories')
    ax1.set_xlabel('u')
    ax1.set_ylabel('v (bottom origin, up)')
    ax1.legend(title='RPM')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 计算统一的坐标轴范围
    if all_x_coords and all_y_coords:
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        
        # 添加一些边距
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        x_range_unified = (x_min - x_margin, x_max + x_margin)
        y_range_unified = (y_min - y_margin, y_max + y_margin)
        
        # 设置原始数据图的坐标轴范围
        ax1.set_xlim(x_range_unified)
        ax1.set_ylim(y_range_unified)
    else:
        x_range_unified = None
        y_range_unified = None

    # 子图2: 预测轨迹对比（使用相同的坐标轴范围）
    if fitted_models:
        # 首先画出所有原始数据作为对比基准
        for rpm in rpm_values:
            df_rpm_all = all_pixels[all_pixels['rpm'] == rpm]
            file_groups = [(fname, grp.copy()) for fname, grp in df_rpm_all.groupby('source_file')]
            
            # 画出所有轨迹，不只是第一条
            for i, (fname, df_file) in enumerate(file_groups):
                if len(df_file) < 20:
                    continue

                df_file.sort_values('timestamp', inplace=True)
                img_h = df_file['ball_v'].max() + 10
                x_vals = df_file['ball_u'].values
                y_vals = img_h - df_file['ball_v'].values

                if len(y_vals) == 0:
                    continue

                idx_min = int(np.argmin(y_vals))
                x_plot = x_vals[:idx_min + 1]
                y_plot = y_vals[:idx_min + 1]

                if len(x_plot) < 20:
                    continue

                # 使用半透明显示多条轨迹，只在第一条上标注label
                label = f'Original RPM {rpm}' if i == 0 else None
                alpha = 0.7 if i == 0 else 0.4  # 第一条更明显
                linewidth = 2 if i == 0 else 1.5
                
                ax2.plot(x_plot, y_plot, color=rpm_color[rpm], linewidth=linewidth, 
                        alpha=alpha, label=label)
        
        # 然后画预测轨迹（完整的抛物线）
        if x_range_unified and y_range_unified:
            # 扩展x范围以显示完整的抛物线
            x_range_extended = (x_range_unified[0] - (x_range_unified[1] - x_range_unified[0]) * 0.5,
                               x_range_unified[1] + (x_range_unified[1] - x_range_unified[0]) * 0.5)
            y_range_extended = (y_range_unified[0] - (y_range_unified[1] - y_range_unified[0]) * 0.3,
                               y_range_unified[1] + (y_range_unified[1] - y_range_unified[0]) * 0.2)
            
            # 使用扩展的x范围来生成预测轨迹
            x_unified = np.linspace(x_range_extended[0], x_range_extended[1], 500)
            
            # 创建更多的预测RPM点，包括原始RPM和插值RPM
            prediction_rpms = list(rpm_values)  # 原始RPM
            
            # 在相邻RPM之间添加插值点
            for i in range(len(rpm_values) - 1):
                mid_rpm = (rpm_values[i] + rpm_values[i + 1]) / 2
                prediction_rpms.append(mid_rpm)
                
                # 在每个区间再添加两个四分位点
                quarter1_rpm = rpm_values[i] + (rpm_values[i + 1] - rpm_values[i]) * 0.25
                quarter3_rpm = rpm_values[i] + (rpm_values[i + 1] - rpm_values[i]) * 0.75
                prediction_rpms.extend([quarter1_rpm, quarter3_rpm])
            
            # 添加一些外推点
            rpm_min, rpm_max = min(rpm_values), max(rpm_values)
            rpm_range = rpm_max - rpm_min
            
            # 向下外推几个点
            prediction_rpms.extend([
                rpm_min - rpm_range * 0.1,
                rpm_min - rpm_range * 0.05
            ])
            
            # 向上外推几个点
            prediction_rpms.extend([
                rpm_max + rpm_range * 0.05,
                rpm_max + rpm_range * 0.1
            ])
            
            # 排序并去重
            prediction_rpms = sorted(list(set(prediction_rpms)))
            
            print(f"预测轨迹将显示 {len(prediction_rpms)} 个RPM值")
            print(f"包括原始RPM: {rpm_values}")
            print(f"以及插值/外推RPM: {[f'{rpm:.1f}' for rpm in prediction_rpms if rpm not in rpm_values]}")
            
            # 为预测RPM创建颜色映射
            prediction_colors = COLOR_MAP(np.linspace(0, 1, len(prediction_rpms)))
            
            for i, rpm in enumerate(prediction_rpms):
                predicted_coeffs = predict_trajectory_coefficients(rpm, fitted_models)
                
                # 计算预测的y值
                predicted_y = np.polyval(predicted_coeffs, x_unified)
                
                # 显示完整的抛物线，但限制在合理的y范围内
                valid_mask = (predicted_y >= y_range_extended[0]) & (predicted_y <= y_range_extended[1])
                
                if np.any(valid_mask):
                    x_valid = x_unified[valid_mask]
                    y_valid = predicted_y[valid_mask]
                    
                    if len(x_valid) > 10:
                        # 找到抛物线的顶点，确保显示完整的弧线
                        a, b, c = predicted_coeffs
                        if a != 0:
                            vertex_x = -b / (2 * a)
                            vertex_y = a * vertex_x**2 + b * vertex_x + c
                            
                            # 确保顶点附近的轨迹都被包含
                            vertex_mask = (x_valid >= vertex_x - abs(vertex_x - x_range_unified[0]) * 0.5) & \
                                         (x_valid <= vertex_x + abs(x_range_unified[1] - vertex_x) * 1.5)
                            
                            if np.any(vertex_mask):
                                x_display = x_valid[vertex_mask]
                                y_display = y_valid[vertex_mask]
                            else:
                                x_display = x_valid
                                y_display = y_valid
                        else:
                            x_display = x_valid
                            y_display = y_valid
                        
                        if len(x_display) > 5:
                            # 区分原始RPM和插值RPM的显示样式
                            if rpm in rpm_values:
                                # 原始RPM：使用对应颜色的粗虚线
                                color = rpm_color[rpm]
                                linewidth = 3
                                alpha = 0.9
                                linestyle = '--'
                                label = f'Predicted RPM {rpm:.0f}'
                            else:
                                # 插值/外推RPM：使用较细的点线
                                color = prediction_colors[i]
                                linewidth = 1.5
                                alpha = 0.6
                                linestyle = ':'
                                label = f'Pred. {rpm:.1f}' if i < 3 else None  # 只为前几个添加标签
                            
                            ax2.plot(x_display, y_display, linestyle, color=color, 
                                    linewidth=linewidth, alpha=alpha, label=label)
        else:
            # 如果没有统一范围，使用原来的方法但扩展范围
            for rpm in rpm_values:
                predicted_coeffs = predict_trajectory_coefficients(rpm, fitted_models)
                
                # 从原始数据中获取该RPM的典型x范围，然后扩展
                df_rpm = all_pixels[all_pixels['rpm'] == rpm]
                if not df_rpm.empty:
                    x_min_orig = df_rpm['ball_u'].min()
                    x_max_orig = df_rpm['ball_u'].max()
                    x_range_width = x_max_orig - x_min_orig
                    
                    # 扩展范围以显示完整抛物线
                    x_min_extended = x_min_orig - x_range_width * 0.5
                    x_max_extended = x_max_orig + x_range_width * 1.0
                    
                    x_range = np.linspace(x_min_extended, x_max_extended, 400)
                    predicted_y = np.polyval(predicted_coeffs, x_range)
                    
                    # 过滤合理的y值
                    y_min_filter = df_rpm['ball_v'].min() - 100
                    y_max_filter = df_rpm['ball_v'].max() + 100
                    
                    valid_mask = (predicted_y >= y_min_filter) & (predicted_y <= y_max_filter)
                    
                    if np.any(valid_mask):
                        x_traj = x_range[valid_mask]
                        y_traj = predicted_y[valid_mask]
                        
                        if len(x_traj) > 10:
                            ax2.plot(x_traj, y_traj, '--', color=rpm_color[rpm], 
                                    linewidth=3, alpha=0.9, label=f'Predicted RPM {rpm}')

    ax2.set_title('Original vs Predicted Trajectories')
    ax2.set_xlabel('u')
    ax2.set_ylabel('v (bottom origin, up)')
    ax2.legend(title='Trajectories', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 为了显示完整的预测轨迹，可能需要调整坐标轴范围
    if x_range_unified and y_range_unified:
        # 稍微扩展坐标轴范围以容纳完整的预测轨迹
        x_expand = (x_range_unified[1] - x_range_unified[0]) * 0.3
        y_expand = (y_range_unified[1] - y_range_unified[0]) * 0.2
        
        ax2.set_xlim(x_range_unified[0] - x_expand, x_range_unified[1] + x_expand)
        ax2.set_ylim(y_range_unified[0] - y_expand, y_range_unified[1] + y_expand)

    # 子图3: 参数拟合结果
    for param_name in ['a', 'b', 'c']:
        if param_name in fitted_models:
            param_array = {'a': a_array, 'b': b_array, 'c': c_array}[param_name]
            best_model_name = fitted_models[param_name]['best_model']
            model_info = fitted_models[param_name]['models'][best_model_name]
            
            # 原始数据点
            ax3.scatter(rpm_array, param_array, label=f'{param_name} data', alpha=0.7, s=50)
            
            # 拟合曲线
            rpm_smooth = np.linspace(min(rpm_array), max(rpm_array), 100)
            param_pred = model_info['model_func'](rpm_smooth, *model_info['params'])
            ax3.plot(rpm_smooth, param_pred, '--', 
                    label=f'{param_name} fit ({best_model_name})', linewidth=2)

    ax3.set_title('Parameter vs RPM')
    ax3.set_xlabel('RPM')
    ax3.set_ylabel('Parameter Value')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 子图4: 模型性能比较
    if fitted_models:
        model_names = []
        r2_scores = []
        
        for param in ['a', 'b', 'c']:
            if param in fitted_models:
                for model_name, model_info in fitted_models[param]['models'].items():
                    model_names.append(f'{param}-{model_name}')
                    r2_scores.append(model_info['r2'])
        
        y_pos = np.arange(len(model_names))
        bars = ax4.barh(y_pos, r2_scores, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(model_names, fontsize=8)
        ax4.set_xlabel('R² Score')
        ax4.set_title('Model Fit Performance')
        ax4.grid(True, axis='x', alpha=0.5)
        
        for i, (lbl, r2) in enumerate(zip(model_names, r2_scores)):
            # 默认基于 R² 的配色
            if r2 > 0.8:
                base_color = 'green'
            elif r2 > 0.6:
                base_color = 'orange'
            else:
                base_color = 'red'

            # 若为参数 b 的模型，统一涂成黄色
            if lbl.startswith('b-'):
                bars[i].set_color('yellow')
            else:
                bars[i].set_color(base_color)

    plt.tight_layout()
    # 移除这里的 plt.show()，避免显示空图表

    # 输出坐标轴范围信息
    if x_range_unified and y_range_unified:
        print(f"\n坐标轴范围信息:")
        print(f"X轴范围: {x_range_unified[0]:.1f} 到 {x_range_unified[1]:.1f}")
        print(f"Y轴范围: {y_range_unified[0]:.1f} 到 {y_range_unified[1]:.1f}")

    # 先输出所有文本信息，再显示图表
    print("\n" + "="*60)
    print("最终拟合函数和预测轨迹方程:")
    print("="*60)
    
    prediction_equations = {}  # 存储预测方程用于图表标注
    
    for param_name in ['a', 'b', 'c']:
        if param_name in fitted_models:
            best_model_name = fitted_models[param_name]['best_model']
            model_info = fitted_models[param_name]['models'][best_model_name]
            params = model_info['params']
            r2 = model_info['r2']
            
            print(f"\n参数 {param_name} 与 RPM 的关系 (R² = {r2:.4f}):")
            print(f"  最佳模型: {best_model_name}")
            print(f"  拟合参数 (高精度): {[f'{p:.12f}' for p in params]}")
            
            if best_model_name == 'Linear':
                equation_str = f"{params[0]:.12f} * rpm + {params[1]:.12f}"
                print(f"  {param_name}(rpm) = {equation_str}")
            elif best_model_name == 'Quadratic':
                equation_str = f"{params[0]:.12f} + {params[1]:.12f} * rpm + {params[2]:.12f} * rpm²"
                print(f"  {param_name}(rpm) = {equation_str}")
            elif best_model_name == 'Cubic':
                equation_str = f"{params[0]:.12f} + {params[1]:.12f} * rpm + {params[2]:.12f} * rpm² + {params[3]:.12f} * rpm³"
                print(f"  {param_name}(rpm) = {equation_str}")
            elif best_model_name == 'Physics':
                equation_str = f"-{params[2]:.12f}/({params[0]:.12f}*rpm + {params[1]:.12f})² + {params[3]:.12f}"
                print(f"  {param_name}(rpm) = {equation_str}")
                print(f"  其中 A={params[0]:.12f}, B={params[1]:.12f}, C={params[2]:.12f}, D={params[3]:.12f}")
            elif best_model_name == 'Exponential':
                equation_str = f"{params[0]:.12f} + {params[1]:.12f} * exp({params[2]:.12f} * rpm)"
                print(f"  {param_name}(rpm) = {equation_str}")
            elif best_model_name == 'Power':
                equation_str = f"{params[0]:.12f} + {params[1]:.12f} * (rpm + 1)^{params[2]:.12f}"
                print(f"  {param_name}(rpm) = {equation_str}")
            else:
                equation_str = f"未知模型: {params}"
                print(f"  参数: {params}")
            
            # 添加验证：显示几个测试点的计算结果
            print(f"  验证计算 (使用当前参数):")
            test_rpms_verify = [min(rpm_array), max(rpm_array)] if len(rpm_array) > 1 else [rpm_array[0]]
            # 特别添加您提到的RPM=3939进行验证
            if 3939 not in test_rpms_verify:
                test_rpms_verify.append(3939)
            
            for test_rpm in test_rpms_verify:
                calculated_value = model_info['model_func'](test_rpm, *params)
                print(f"    RPM={test_rpm:.0f} -> {param_name}={calculated_value:.12f}")
                
                # 如果是RPM=3939，额外显示计算步骤
                if test_rpm == 3939:
                    print(f"    RPM=3939 详细计算步骤:")
                    if best_model_name == 'Cubic':
                        a0, a1, a2, a3 = params
                        step1 = a0
                        step2 = a1 * test_rpm
                        step3 = a2 * test_rpm**2
                        step4 = a3 * test_rpm**3
                        total = step1 + step2 + step3 + step4
                        print(f"      {a0:.12f} + {a1:.12f}*{test_rpm} + {a2:.12f}*{test_rpm}² + {a3:.12f}*{test_rpm}³")
                        print(f"      = {step1:.12f} + {step2:.12f} + {step3:.12f} + {step4:.12f}")
                        print(f"      = {total:.12f}")
                    elif best_model_name == 'Linear':
                        a, b = params
                        result = a * test_rpm + b
                        print(f"      {a:.12f}*{test_rpm} + {b:.12f} = {result:.12f}")
                    elif best_model_name == 'Quadratic':
                        a0, a1, a2 = params
                        step1 = a0
                        step2 = a1 * test_rpm
                        step3 = a2 * test_rpm**2
                        total = step1 + step2 + step3
                        print(f"      {a0:.12f} + {a1:.12f}*{test_rpm} + {a2:.12f}*{test_rpm}²")
                        print(f"      = {step1:.12f} + {step2:.12f} + {step3:.12f}")
                        print(f"      = {total:.12f}")

    print(f"\n" + "-"*60)
    print("完整的轨迹预测系统:")
    print("-"*60)
    print("给定任意 RPM 值，轨迹方程为:")
    print("y = a(rpm) * x² + b(rpm) * x + c(rpm)")
    print("\n其中各参数函数为:")
    
    for param_name in ['a', 'b', 'c']:
        if param_name in prediction_equations:
            eq_info = prediction_equations[param_name]
            print(f"  {param_name}(rpm) = {eq_info['equation']}")
            print(f"    (使用 {eq_info['model']} 模型, R² = {eq_info['r2']:.4f})")

    # 给出几个具体示例预测
    if fitted_models and rpm_array.size > 0:
        print(f"\n" + "-"*60)
        print("具体示例预测:")
        print("-"*60)
        
        # 扩展预测RPM范围，包括原始数据范围内外的测试点
        rpm_min, rpm_max = min(rpm_array), max(rpm_array)
        rpm_range = rpm_max - rpm_min
        
        # 生成更全面的测试RPM列表
        test_rpms = []
        
        # 1. 原始数据范围内的测试点
        test_rpms.extend([
            rpm_min,  # 最小值
            rpm_min + rpm_range * 0.25,  # 第一四分位
            rpm_min + rpm_range * 0.5,   # 中位数
            rpm_min + rpm_range * 0.75,  # 第三四分位
            rpm_max   # 最大值
        ])
        
        # 2. 原始数据范围外的外推测试点
        test_rpms.extend([
            rpm_min - rpm_range * 0.2,  # 下外推20%
            rpm_min - rpm_range * 0.1,  # 下外推10%
            rpm_max + rpm_range * 0.1,  # 上外推10%
            rpm_max + rpm_range * 0.2   # 上外推20%
        ])
        
        # 3. 特定的测试点
        specific_test_rpms = [3939, 4000, 5000, 6000]  # 包括您提到的3939
        for rpm in specific_test_rpms:
            if rpm not in test_rpms:
                test_rpms.append(rpm)
        
        # 4. 一些常见的整数RPM值
        common_rpms = [3000, 3500, 4500, 5500, 6500, 7000]
        for rpm in common_rpms:
            if rpm_min <= rpm <= rpm_max and rpm not in test_rpms:
                test_rpms.append(rpm)
        
        # 排序并去重
        test_rpms = sorted(list(set(test_rpms)))
        
        print(f"原始数据RPM范围: {rpm_min:.0f} - {rpm_max:.0f}")
        print(f"测试{len(test_rpms)}个RPM值 (包括范围内插值和范围外外推):")
        print(f"测试RPM列表: {[f'{rpm:.0f}' for rpm in test_rpms]}")
        
        for i, test_rpm in enumerate(test_rpms):
            coeffs = predict_trajectory_coefficients(test_rpm, fitted_models)
            
            # 标识数据类型
            if test_rpm < rpm_min:
                data_type = "外推(下)"
            elif test_rpm > rpm_max:
                data_type = "外推(上)"
            else:
                data_type = "内插"
            
            print(f"\n[{i+1:2d}/{len(test_rpms)}] RPM = {test_rpm:.0f} ({data_type}):")
            
            # 详细显示计算过程
            print(f"  计算过程 (高精度):")
            for j, param_name in enumerate(['a', 'b', 'c']):
                if param_name in fitted_models:
                    best_model_name = fitted_models[param_name]['best_model']
                    model_info = fitted_models[param_name]['models'][best_model_name]
                    param_value = model_info['model_func'](test_rpm, *model_info['params'])
                    print(f"    {param_name}({test_rpm:.0f}) = {param_value:.12f}")
                    
                    # 验证coeffs数组中的值是否匹配
                    if j < len(coeffs):
                        match_status = '✓' if abs(coeffs[j] - param_value) < 1e-12 else '✗'
                        print(f"    验证: coeffs[{j}] = {coeffs[j]:.12f} {match_status}")
                        if abs(coeffs[j] - param_value) >= 1e-12:
                            print(f"    差异: {abs(coeffs[j] - param_value):.15e}")
            
            print(f"  轨迹方程: y = {coeffs[0]:.12f}*x² + {coeffs[1]:.12f}*x + {coeffs[2]:.12f}")
            
            # 计算并显示轨迹特征
            a, b, c = coeffs
            if abs(a) > 1e-12:  # 非退化抛物线
                vertex_x = -b / (2 * a)
                vertex_y = a * vertex_x**2 + b * vertex_x + c
                print(f"  轨迹特征: {'开口向下' if a < 0 else '开口向上'}, 顶点({vertex_x:.1f}, {vertex_y:.1f})")
                
                # 物理合理性检查
                if a > 0:
                    print(f"  ⚠️  物理警告: 开口向上的抛物线不符合重力作用下的弹道轨迹")
                else:
                    print(f"  ✓  物理检查: 开口向下，符合重力弹道轨迹")
            else:
                print(f"  轨迹特征: 退化为直线 (a ≈ 0)")
            
            # 特别标注重要的测试点
            if abs(test_rpm - 3939) < 0.1:
                print(f"  *** 这是您特别关注的RPM=3939测试点 ***")
            
            if data_type.startswith("外推"):
                print(f"  注意: 这是范围外{data_type}，预测可靠性可能降低")

    # 添加特别的RPM=3939验证
    if fitted_models:
        print(f"\n" + "-"*60)
        print("特别验证：RPM=3939 的计算")
        print("-"*60)
        
        rpm_test = 3939
        coeffs_3939 = predict_trajectory_coefficients(rpm_test, fitted_models)
        
        print(f"使用完整精度计算 RPM={rpm_test} 时的轨迹参数:")
        for i, param_name in enumerate(['a', 'b', 'c']):
            if param_name in fitted_models:
                best_model_name = fitted_models[param_name]['best_model']
                model_info = fitted_models[param_name]['models'][best_model_name]
                params = model_info['params']
                
                print(f"\n参数 {param_name}:")
                print(f"  模型: {best_model_name}")
                print(f"  系数: {[f'{p:.15f}' for p in params]}")
                
                # 手动计算
                if best_model_name == 'Cubic':
                    manual_calc = params[0] + params[1]*rpm_test + params[2]*rpm_test**2 + params[3]*rpm_test**3
                elif best_model_name == 'Linear':
                    manual_calc = params[0]*rpm_test + params[1]
                elif best_model_name == 'Quadratic':
                    manual_calc = params[0] + params[1]*rpm_test + params[2]*rpm_test**2
                else:
                    manual_calc = model_info['model_func'](rpm_test, *params)
                
                func_calc = model_info['model_func'](rpm_test, *params)
                
                print(f"  手动计算: {manual_calc:.15f}")
                print(f"  函数计算: {func_calc:.15f}")
                print(f"  coeffs[{i}]: {coeffs_3939[i]:.15f}")
                print(f"  差异: {abs(manual_calc - func_calc):.15e}")
        
        print(f"\n最终结果：RPM=3939 时，a = {coeffs_3939[0]:.15f}")
        print(f"开口方向：{'向上 (a>0)' if coeffs_3939[0] > 0 else '向下 (a<0)' if coeffs_3939[0] < 0 else '退化为直线 (a=0)'}")

    print(f"\n" + "="*60)
    print("虚线绘制的技术细节:")
    print("="*60)
    print("虚线是通过以下步骤绘制的:")
    print("1. 调用 predict_trajectory_coefficients(rpm, fitted_models)")
    print("2. 该函数返回 [a, b, c] 系数数组")
    print("3. 使用 np.polyval(coeffs, x_range) 计算 y = ax² + bx + c")
    print("4. 绘制 ax2.plot(x_range, predicted_y, '--', ...)")
    print("\n如果虚线正确但输出的函数错误，说明:")
    print("- predict_trajectory_coefficients() 函数工作正常")
    print("- 文本输出的参数顺序或格式可能有误")
    print("- 建议查看上方的验证计算结果确认参数值")

    print(f"\n" + "="*60)
    print("所有文本输出完成，正在显示图表...")
    print("关闭图表窗口后程序将结束。")
    print("="*60)

    # 确保图表布局正确，然后显示
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()