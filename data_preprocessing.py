import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from scipy.stats import entropy
from scipy.signal import find_peaks
from scipy.fft import fft
from tqdm import tqdm
import warnings
import joblib
from datetime import datetime

# 抑制所有警告
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*mean of empty slice.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')

class RadarDataProcessor:
    """
    雷达数据处理器
    处理包含 FrameNumber, ObjectNumber, Range, Velocity, PeakValue, x, y 的数据
    """
    def __init__(self, data_dir, output_dir, n_components=50):
        """
        初始化数据处理器
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            n_components: PCA保留的特征数量
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def _safe_compute(self, func, default_value=0):
        """
        安全计算函数，处理可能的异常情况
        """
        try:
            result = func()
            if np.isnan(result) or np.isinf(result):
                return default_value
            return result
        except:
            return default_value
    
    def extract_radar_features(self, df):
        """
        从雷达数据中提取特征
        Args:
            df: 包含单个手势的DataFrame
        Returns:
            features: 提取的特征向量
        """
        # 基础统计特征
        range_features = self._compute_statistical_features(df['Range'])
        velocity_features = self._compute_statistical_features(df['Velocity'])
        peak_features = self._compute_statistical_features(df['PeakValue'])
        
        # 轨迹特征
        trajectory_features = self._compute_trajectory_features(df[['x', 'y']])
        
        # 时序特征
        temporal_features = self._compute_temporal_features(df)
        
        # FFT特征
        fft_features = self._compute_fft_features(df)
        
        # 点云特征
        pointcloud_features = self._compute_pointcloud_features(df)
        
        # 相关性特征
        correlation_features = self._compute_correlation_features(df)
        
        # 动态特征
        dynamic_features = self._compute_dynamic_features(df)
        
        # 组合所有特征
        features = np.concatenate([
            range_features,        # 10维
            velocity_features,     # 10维
            peak_features,        # 10维
            trajectory_features,   # 15维
            temporal_features,     # 10维
            fft_features,         # 30维
            pointcloud_features,   # 15维
            correlation_features,  # 6维
            dynamic_features      # 12维
        ])
        
        return features
    
    def _compute_statistical_features(self, series):
        """
        计算统计特征，处理异常值
        """
        # 先进行插值，再处理异常值
        clean_series = series.replace([np.inf, -np.inf], np.nan)
        clean_series = clean_series.interpolate(method='linear', limit_direction='both')
        clean_series = clean_series.fillna(clean_series.mean() if not pd.isna(clean_series.mean()) else 0)
        
        if len(clean_series) == 0:
            return np.zeros(10)
            
        features = []
        # 基本统计量
        features.append(float(clean_series.mean()))
        features.append(float(clean_series.std()) if len(clean_series) > 1 else 0)
        features.append(float(clean_series.skew()) if len(clean_series) > 2 else 0)
        features.append(float(clean_series.kurtosis()) if len(clean_series) > 3 else 0)
        features.append(float(clean_series.max()))
        features.append(float(clean_series.min()))
        
        # 分位数统计
        quantiles = clean_series.quantile([0.25, 0.5, 0.75])
        features.extend([
            float(quantiles[0.25]),
            float(quantiles[0.75]),
            float(quantiles[0.5]),  # median
            float(clean_series.max() - clean_series.min())  # range
        ])
        
        # 确保所有特征都是有效的数值
        features = [0 if np.isnan(x) or np.isinf(x) else x for x in features]
        return np.array(features)
    
    def _compute_trajectory_features(self, trajectory_df):
        """
        计算增强的轨迹特征
        """
        # 基础轨迹特征
        diff = trajectory_df.diff()
        distances = np.sqrt(diff['x']**2 + diff['y']**2)
        total_distance = distances.sum()
        
        start_point = trajectory_df.iloc[0]
        end_point = trajectory_df.iloc[-1]
        direct_distance = np.sqrt(
            (end_point['x'] - start_point['x'])**2 + 
            (end_point['y'] - start_point['y'])**2
        )
        
        # 计算轨迹的曲率
        curvature = total_distance / (direct_distance + 1e-6)
        
        # 计算轨迹的方向变化
        angles = np.arctan2(diff['y'], diff['x'])
        angle_changes = np.abs(angles.diff()).sum()
        
        # 计算轨迹的边界框特征
        bbox_width = trajectory_df['x'].max() - trajectory_df['x'].min()
        bbox_height = trajectory_df['y'].max() - trajectory_df['y'].min()
        bbox_area = bbox_width * bbox_height
        
        # 计算轨迹的速度特征
        velocities = distances / trajectory_df.index.to_series().diff()
        avg_velocity = velocities.mean()
        max_velocity = velocities.max()
        velocity_std = velocities.std()
        
        # 计算轨迹的加速度特征
        accelerations = velocities.diff()
        avg_acceleration = accelerations.mean()
        max_acceleration = accelerations.max()
        acceleration_std = accelerations.std()
        
        features = [
            total_distance,      # 总路程
            direct_distance,     # 直线距离
            curvature,          # 曲率
            angle_changes,       # 方向变化总量
            bbox_width,         # 边界框宽度
            bbox_height,        # 边界框高度
            bbox_area,          # 边界框面积
            avg_velocity,       # 平均速度
            max_velocity,       # 最大速度
            velocity_std,       # 速度标准差
            avg_acceleration,   # 平均加速度
            max_acceleration,   # 最大加速度
            acceleration_std,   # 加速度标准差
            bbox_width/bbox_height,  # 宽高比
            total_distance/direct_distance  # 路径效率
        ]
        return np.array(features)
    
    def _compute_temporal_features(self, df):
        """
        计算增强的时序特征
        """
        # 确保列名已经清理过空格
        df.columns = df.columns.str.strip()
        
        # 基础时序变化
        velocity_changes = df['Velocity'].diff().abs()
        range_changes = df['Range'].diff().abs()
        peak_changes = df['PeakValue'].diff().abs()
        
        features = [
            velocity_changes.mean(),  # 平均速度变化
            velocity_changes.max(),   # 最大速度变化
            velocity_changes.std(),   # 速度变化标准差
            range_changes.mean(),     # 平均距离变化
            range_changes.max(),      # 最大距离变化
            range_changes.std(),      # 距离变化标准差
            peak_changes.mean(),      # 平均信号强度变化
            peak_changes.max(),       # 最大信号强度变化
            peak_changes.std(),       # 信号强度变化标准差
            df['ObjectNumber'].nunique()  # 目标数量
        ]
        return np.array(features)
    
    def _compute_pointcloud_features(self, df):
        """
        计算点云特征
        """
        # 确保列名已经清理过空格
        df.columns = df.columns.str.strip()
        
        # 计算点云密度
        points = df[['x', 'y']].values
        
        # 计算点云的协方差矩阵
        cov_matrix = np.cov(points.T)
        if cov_matrix.shape == (2, 2):
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigval_ratio = min(eigenvalues) / (max(eigenvalues) + 1e-6)
        else:
            eigval_ratio = 0
            
        # 计算点云的分布特征
        centroid = points.mean(axis=0)
        distances_to_centroid = np.sqrt(np.sum((points - centroid)**2, axis=1))
        
        # 计算每个时间步的点数
        points_per_frame = df.groupby('FrameNumber').size()
        
        # 计算点云的空间分布
        x_spread = df['x'].max() - df['x'].min()
        y_spread = df['y'].max() - df['y'].min()
        area = x_spread * y_spread
        point_density = len(df) / (area + 1e-6)
        
        # 计算点云的形状特征
        hull_area = self._compute_convex_hull_area(points)
        compactness = hull_area / (area + 1e-6)
        
        features = [
            point_density,                    # 点密度
            eigval_ratio,                    # 特征值比（反映点云的方向性）
            distances_to_centroid.mean(),     # 到中心点的平均距离
            distances_to_centroid.std(),      # 到中心点的距离标准差
            points_per_frame.mean(),          # 平均每帧点数
            points_per_frame.std(),           # 每帧点数的标准差
            points_per_frame.max(),           # 最大帧点数
            points_per_frame.min(),           # 最小帧点数
            x_spread,                         # x方向跨度
            y_spread,                         # y方向跨度
            area,                             # 点云覆盖面积
            hull_area,                        # 凸包面积
            compactness,                      # 紧凑度
            df['ObjectNumber'].mean(),        # 平均目标数
            df['ObjectNumber'].std()          # 目标数标准差
        ]
        return np.array(features)
        
    def _compute_convex_hull_area(self, points):
        """
        计算点云的凸包面积
        """
        try:
            from scipy.spatial import ConvexHull
            if len(points) >= 3:
                hull = ConvexHull(points)
                return hull.area
            return 0
        except:
            return 0
    
    def _compute_correlation_features(self, df):
        """
        计算相关性特征
        """
        # 确保列名已经清理过空格
        df.columns = df.columns.str.strip()
        
        # 计算主要信号之间的相关性
        corr_matrix = df[['Range', 'Velocity', 'PeakValue']].corr()
        
        features = [
            corr_matrix.loc['Range', 'Velocity'],     # Range-Velocity相关性
            corr_matrix.loc['Range', 'PeakValue'],    # Range-PeakValue相关性
            corr_matrix.loc['Velocity', 'PeakValue'], # Velocity-PeakValue相关性
            # 计算x,y与其他特征的相关性
            df['x'].corr(df['Velocity']),            # x-Velocity相关性
            df['y'].corr(df['Velocity']),            # y-Velocity相关性
            df['PeakValue'].corr(df['ObjectNumber']) # PeakValue-ObjectNumber相关性
        ]
        return np.array(features)
    
    def _compute_dynamic_features(self, df):
        """
        计算动态特征，替代原来的频域特征
        """
        # 计算速度方向变化
        velocity_direction = np.sign(df['Velocity'])
        direction_changes = np.abs(np.diff(velocity_direction)).sum()
        
        # 计算加速度特征
        acceleration = df['Velocity'].diff() / df['FrameNumber'].diff()
        
        # 计算Range变化率
        range_rate = df['Range'].diff() / df['FrameNumber'].diff()
        
        # 计算PeakValue变化率
        peak_rate = df['PeakValue'].diff() / df['FrameNumber'].diff()
        
        # 计算运动阶段
        motion_phases = self._identify_motion_phases(df['Velocity'])
        
        features = [
            direction_changes,                    # 速度方向变化次数
            acceleration.mean(),                  # 平均加速度
            acceleration.std(),                   # 加速度标准差
            acceleration.max(),                   # 最大加速度
            acceleration.min(),                   # 最小加速度
            range_rate.mean(),                   # 平均距离变化率
            range_rate.std(),                    # 距离变化率标准差
            peak_rate.mean(),                    # 平均信号强度变化率
            peak_rate.std(),                     # 信号强度变化率标准差
            motion_phases['acceleration_phase'],  # 加速阶段占比
            motion_phases['stable_phase'],       # 稳定阶段占比
            motion_phases['deceleration_phase']  # 减速阶段占比
        ]
        return np.array(features)
        
    def _identify_motion_phases(self, velocity):
        """
        识别运动的不同阶段
        """
        # 计算加速度
        accel = velocity.diff()
        
        # 定义阈值
        threshold = 0.1 * accel.std()
        
        # 识别不同阶段
        acceleration_phase = np.sum(accel > threshold)
        deceleration_phase = np.sum(accel < -threshold)
        stable_phase = np.sum(np.abs(accel) <= threshold)
        
        total_frames = len(velocity)
        
        return {
            'acceleration_phase': acceleration_phase / total_frames,
            'stable_phase': stable_phase / total_frames,
            'deceleration_phase': deceleration_phase / total_frames
        }
    
    def _compute_fft_features(self, df):
        """
        计算FFT特征，处理异常值
        """
        features = []
        for col in ['Range', 'Velocity', 'PeakValue']:
            # 数据预处理
            signal = df[col].replace([np.inf, -np.inf], np.nan)
            signal = signal.interpolate(method='linear', limit_direction='both')
            signal = signal.fillna(signal.mean() if not pd.isna(signal.mean()) else 0)
            signal = signal.values
            
            # 去均值
            signal = signal - np.mean(signal)
            
            # 对信号进行FFT
            fft_vals = np.abs(fft(signal))[:len(signal)//2]
            n_components = min(10, len(fft_vals))
            fft_main = fft_vals[:n_components]
            
            # 计算频谱特征
            fft_sum = np.sum(fft_main)
            if fft_sum < 1e-10:
                features.extend(np.zeros(10))
                continue
                
            fft_norm = fft_main / (fft_sum + 1e-10)
            
            # 计算特征并确保它们是有效的
            curr_features = [
                float(np.mean(fft_norm)),
                float(np.std(fft_norm)) if len(fft_norm) > 1 else 0,
                float(entropy(fft_norm + 1e-10)),
                float(np.max(fft_norm)),
                float(np.argmax(fft_norm)),
                float(np.sum(fft_norm[:3])) if len(fft_norm) >= 3 else 0,
                float(np.sum(fft_norm[-3:])) if len(fft_norm) >= 3 else 0,
                float(np.sum(fft_norm[3:-3])) if len(fft_norm) >= 7 else 0,
                float(np.percentile(fft_norm, 25)),
                float(np.percentile(fft_norm, 75))
            ]
            
            # 确保所有特征都是有效的数值
            curr_features = [0 if np.isnan(x) or np.isinf(x) else x for x in curr_features]
            features.extend(curr_features)
        
        return np.array(features)
    
    def process_dataset(self):
        """
        处理整个数据集
        """
        features_list = []
        labels = []
        
        # 创建日志目录
        log_dir = os.path.join(self.output_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建处理日志文件
        processing_log = []
        class_stats = []
        
        # 记录处理统计信息
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'empty_features': 0,
            'invalid_features': 0
        }
        
        # 获取所有类别
        class_folders = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # 使用tqdm显示总体进度
        for class_name in tqdm(class_folders, desc="处理数据集", ncols=100):
            class_path = os.path.join(self.data_dir, class_name)
            class_features = []
            
            # 获取当前类别的所有文件
            csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
            stats['total_files'] += len(csv_files)
            
            class_log = {
                '类别': class_name,
                '文件总数': len(csv_files),
                '处理成功': 0,
                '处理失败': 0,
                '空特征': 0,
                '无效特征': 0
            }
            
            # 处理当前类别的所有文件
            for file_name in csv_files:
                file_path = os.path.join(class_path, file_name)
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                try:
                    # 读取CSV文件
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip()
                    
                    # 检查必需的列
                    required_columns = ['FrameNumber', 'ObjectNumber', 'Range', 'Velocity', 'PeakValue', 'x', 'y']
                    if not all(col in df.columns for col in required_columns):
                        stats['failed_files'] += 1
                        class_log['处理失败'] += 1
                        processing_log.append({
                            '时间': current_time,
                            '类别': class_name,
                            '文件': file_name,
                            '状态': '失败',
                            '原因': '缺少必需列'
                        })
                        continue
                    
                    # 数据清理和预处理
                    df = df.replace([np.inf, -np.inf], np.nan)
                    
                    # 计算每列的NaN比例
                    nan_ratio = df[required_columns].isna().mean()
                    
                    # 如果任何必需列的NaN比例超过50%，跳过该样本（放宽阈值）
                    if (nan_ratio > 0.5).any():
                        stats['failed_files'] += 1
                        class_log['处理失败'] += 1
                        processing_log.append({
                            '时间': current_time,
                            '类别': class_name,
                            '文件': file_name,
                            '状态': '失败',
                            '原因': 'NaN值比例过高'
                        })
                        continue
                    
                    # 对不同类型的数据使用不同的插值方法
                    # 时序数据使用线性插值
                    for col in ['Range', 'Velocity', 'PeakValue']:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                        # 对于插值后仍然存在的NaN，使用均值填充
                        mean_val = df[col].mean()
                        if pd.isna(mean_val):
                            mean_val = 0
                        df[col] = df[col].fillna(mean_val)
                    
                    # 空间坐标使用最近邻插值
                    for col in ['x', 'y']:
                        df[col] = df[col].interpolate(method='nearest', limit_direction='both')
                        # 对于插值后仍然存在的NaN，使用均值填充
                        mean_val = df[col].mean()
                        if pd.isna(mean_val):
                            mean_val = 0
                        df[col] = df[col].fillna(mean_val)
                    
                    # ObjectNumber使用前向填充和后向填充
                    df['ObjectNumber'] = df['ObjectNumber'].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
                    # 特征提取
                    try:
                        features = self.extract_radar_features(df)
                        
                        # 检查特征的有效性
                        if len(features) == 0:
                            stats['empty_features'] += 1
                            class_log['空特征'] += 1
                            continue
                        
                        # 检查是否有无效值，并尝试修复
                        invalid_mask = np.isnan(features) | np.isinf(features)
                        if np.any(invalid_mask):
                            # 将无效值替换为0
                            features[invalid_mask] = 0
                        
                        class_features.append(features)
                        stats['processed_files'] += 1
                        class_log['处理成功'] += 1
                        processing_log.append({
                            '时间': current_time,
                            '类别': class_name,
                            '文件': file_name,
                            '状态': '成功',
                            '原因': '处理完成'
                        })
                        
                    except Exception as e:
                        stats['failed_files'] += 1
                        class_log['处理失败'] += 1
                        processing_log.append({
                            '时间': current_time,
                            '类别': class_name,
                            '文件': file_name,
                            '状态': '失败',
                            '原因': f'特征提取错误: {str(e)}'
                        })
                        continue
                    
                except Exception as e:
                    stats['failed_files'] += 1
                    class_log['处理失败'] += 1
                    processing_log.append({
                        '时间': current_time,
                        '类别': class_name,
                        '文件': file_name,
                        '状态': '失败',
                        '原因': f'文件读取错误: {str(e)}'
                    })
                    continue
            
            if len(class_features) > 0:
                features_list.extend(class_features)
                labels.extend([class_name] * len(class_features))
            
            class_stats.append(class_log)
        
        # 保存处理日志
        pd.DataFrame(processing_log).to_csv(os.path.join(log_dir, 'processing_log.csv'), index=False, encoding='utf-8-sig')
        pd.DataFrame(class_stats).to_csv(os.path.join(log_dir, 'class_statistics.csv'), index=False, encoding='utf-8-sig')
        
        if len(features_list) == 0:
            raise ValueError("没有有效的特征数据可供处理")
        
        # 数据处理
        X = np.array(features_list)
        y = np.array(labels)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA降维
        X_pca = self.pca.fit_transform(X_scaled)
        
        # 类别平衡（只在类别差异大于1.5倍时使用SMOTE）
        class_counts = pd.Series(y).value_counts()
        if len(class_counts) > 1 and (class_counts.max() / class_counts.min()) > 1.5:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min()-1))
                X_resampled, y_resampled = smote.fit_resample(X_pca, y)
            except Exception as e:
                print(f"SMOTE处理失败: {str(e)}")
                X_resampled, y_resampled = X_pca, y
        else:
            X_resampled, y_resampled = X_pca, y
        
        # 保存结果
        np.save(os.path.join(self.output_dir, 'features.npy'), X_resampled)
        np.save(os.path.join(self.output_dir, 'labels.npy'), y_resampled)
        joblib.dump(scaler, os.path.join(self.output_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(self.output_dir, 'pca.pkl'))
        
        # 保存处理结果统计
        results = {
            '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '原始特征维度': X.shape,
            'PCA降维后维度': X_pca.shape,
            '最终数据维度': X_resampled.shape,
            'PCA累计方差比': list(np.cumsum(self.pca.explained_variance_ratio_)[:10]),
            '原始类别分布': dict(pd.Series(y).value_counts()),
            '处理后类别分布': dict(pd.Series(y_resampled).value_counts())
        }
        
        pd.DataFrame([results]).to_csv(os.path.join(log_dir, 'processing_results.csv'), index=False, encoding='utf-8-sig')
        
        return X_resampled, y_resampled

if __name__ == '__main__':
    data_dir = 'data'
    output_dir = 'processed_data'
    
    processor = RadarDataProcessor(data_dir, output_dir)
    X_resampled, y_resampled = processor.process_dataset()
    print(f"处理后的数据集大小: {X_resampled.shape}")
    print(f"类别分布:\n{pd.Series(y_resampled).value_counts()}") 
