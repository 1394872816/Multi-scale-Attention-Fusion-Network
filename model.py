import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleAttention(nn.Module):
    """多尺度注意力模块
    
    输入维度: [batch_size, seq_len, dim]
    输出维度: [batch_size, seq_len, dim]
    
    参数:
        dim (int): 输入特征维度，必须能被num_heads整除
        num_heads (int): 注意力头数，默认8
        dropout (float): dropout比率，默认0.1
        
    计算过程:
    1. 输入特征经过线性变换得到Q、K、V，每个形状为 [batch_size, num_heads, seq_len, head_dim]
    2. 计算注意力分数: (Q @ K^T) * scale
    3. Softmax归一化后与V相乘得到注意力特征
    4. 投影回原始维度
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        
        # 多尺度特征提取
        self.scale_factors = [1.0, 0.75, 0.5]
        
        # 增强型特征投影
        self.feature_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(len(self.scale_factors))
        ])
        
        # 通道注意力
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.LayerNorm(dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid()
            ) for _ in range(len(self.scale_factors))
        ])
        
        # 空间注意力
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.LayerNorm(dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(len(self.scale_factors))
        ])
        
        # QKV投影
        self.qkv = nn.ModuleList([
            nn.Linear(dim, dim * 3) for _ in range(len(self.scale_factors))
        ])
        
        # 特征增强
        self.feature_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim)
            ) for _ in range(len(self.scale_factors))
        ])
        
        # 自适应特征聚合
        self.feature_weights = nn.Sequential(
            nn.Linear(dim * len(self.scale_factors), len(self.scale_factors)),
            nn.Softmax(dim=-1)
        )
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.attn_drop = nn.Dropout(dropout)
        # 将 scale 改为注册的缓冲区
        self.register_buffer('scale', torch.tensor(self.head_dim ** -0.5))
        
    def forward(self, x):
        B, N, C = x.shape
        identity = x
        
        # 多尺度特征提取
        multi_scale_features = []
        for i, scale in enumerate(self.scale_factors):
            # 计算当前尺度的序列长度
            H = max(4, int(N * scale))  # 确保序列长度至少为4
            
            # 特征缩放
            if scale != 1.0:
                x_scaled = F.interpolate(
                    x.transpose(1, 2), 
                    size=H, 
                    mode='linear'
                ).transpose(1, 2)
            else:
                x_scaled = x
                
            # 特征投影
            x_scaled = self.feature_proj[i](x_scaled)
            
            # 通道注意力
            channel_weights = self.channel_attention[i](x_scaled.mean(dim=1, keepdim=True))
            x_scaled = x_scaled * channel_weights
            
            # 空间注意力
            spatial_weights = self.spatial_attention[i](x_scaled)
            x_scaled = x_scaled * spatial_weights
            
            # QKV注意力
            qkv = self.qkv[i](x_scaled).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            # 计算注意力分数
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # 相对位置编码
            position_bias = self._get_relative_position_bias(H)
            attn = attn + position_bias
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # 注意力特征
            x_scaled = (attn @ v).transpose(1, 2).reshape(B, -1, C)
            
            # 特征增强
            x_scaled = self.feature_enhance[i](x_scaled)
            
            # 还原原始序列长度
            if scale != 1.0:
                x_scaled = F.interpolate(
                    x_scaled.transpose(1, 2), 
                    size=N, 
                    mode='linear'
                ).transpose(1, 2)
            
            multi_scale_features.append(x_scaled)
        
        # 计算特征权重
        concat_features = torch.cat([f.mean(dim=1) for f in multi_scale_features], dim=-1)
        weights = self.feature_weights(concat_features)  # [B, num_scales]
        
        # 加权融合特征
        x = torch.stack(multi_scale_features, dim=1)  # [B, num_scales, N, C]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_scales, 1, 1]
        x = (x * weights).sum(dim=1)  # [B, N, C]
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 残差连接
        return x + identity
        
    def _get_relative_position_bias(self, seq_len):
        """计算相对位置编码"""
        position_ids = torch.arange(seq_len, device=self.scale.device)
        relative_position = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)
        relative_position = relative_position.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        return relative_position * 0.02  # 缩放因子

class SEAttention(nn.Module):
    """SE (Squeeze-and-Excitation) 注意力模块
    
    输入维度: [batch_size, in_channels]
    输出维度: [batch_size, in_channels]
    
    参数:
        in_channels (int): 输入特征通道数
        reduction (int): 特征降维比例，用于控制参数量，默认8
        
    计算过程:
    1. 双路特征提取：最大池化和平均池化 -> [batch_size, in_channels]
    2. 特征拼接后降维: in_channels*2 -> in_channels//reduction
    3. 升维+归一化: in_channels//reduction -> in_channels
    4. 通道重标定
    """
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.size()
        
        avg_feat = self.avg_pool(x.unsqueeze(-1)).view(batch_size, channels)
        max_feat = self.max_pool(x.unsqueeze(-1)).view(batch_size, channels)
        
        feat = torch.cat([avg_feat, max_feat], dim=1)
        channel_weights = self.mlp(feat).view(batch_size, channels)
        
        return x * channel_weights

class ResidualBlock(nn.Module):
    """残差块
    
    输入维度: [batch_size, dim]
    输出维度: [batch_size, dim]
    
    参数:
        dim (int): 特征维度，输入输出维度相同
        
    计算过程:
    1. 两层线性变换: dim -> dim -> dim
    2. 每层后接LayerNorm和ReLU
    3. 残差连接: output = input + F(input)
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑损失函数
    
    输入维度: 
        pred: [batch_size, num_classes] - 预测logits
        target: [batch_size] - 真实标签索引
    输出维度: 标量损失值
    
    参数:
        smoothing (float): 平滑因子，默认0.1
        
    计算过程:
    1. 预测logits经过log_softmax
    2. 构建平滑标签: (1-smoothing)给真实类别，smoothing/(num_classes-1)给其他类别
    3. 计算交叉熵损失
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class FeatureFusion(nn.Module):
    """特征融合模块
    
    输入维度: [batch_size, 50] - 原始特征
    输出维度: [batch_size, feature_dim] - 融合后的特征
    
    特征分组:
        part1: 0-8   (9维)
        part2: 9-16  (8维)
        part3: 17-24 (8维)
        part4: 25-32 (8维)
        part5: 33-40 (8维)
        part6: 41-49 (9维)
    
    参数:
        feature_dim (int): 特征转换后的维度，默认128
        dropout_rate (float): Dropout比率，默认0.3
        
    计算过程:
    1. 特征分割: 50维分成6组
    2. 特征转换: 每组特征转换到feature_dim维
    3. 特征交互: 6组特征通过MultiScaleAttention交互
    4. 自适应特征聚合
    5. 注意力引导的特征校准
    6. 特征增强
    7. 特征聚合
    8. 特征重标定
    9. 特征聚合
    """
    def __init__(self, feature_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        
        # 特征转换层
        self.feature_transforms = nn.ModuleDict({
            'part1': self._make_transform_layer(9, feature_dim),   # 0-8
            'part2': self._make_transform_layer(8, feature_dim),   # 9-16
            'part3': self._make_transform_layer(8, feature_dim),   # 17-24
            'part4': self._make_transform_layer(8, feature_dim),   # 25-32
            'part5': self._make_transform_layer(8, feature_dim),   # 33-40
            'part6': self._make_transform_layer(9, feature_dim)    # 41-49
        })
        
        # 特征交互
        self.multi_scale_attention = MultiScaleAttention(
            dim=feature_dim,
            num_heads=8,
            dropout=dropout_rate
        )
        
        # 动态特征权重
        self.dynamic_weights = nn.Sequential(
            nn.Linear(feature_dim * 6, 6),
            nn.LayerNorm(6),
            nn.Softmax(dim=-1)
        )
        
        # 特征对比学习投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        # 特征组注意力
        self.group_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # 特征增强
        self.feature_enhancement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # 特征校准
        self.feature_calibration = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # 特征聚合
        self.feature_aggregation = nn.Sequential(
            ResidualBlock(feature_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(feature_dim)
        )
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(feature_dim)
        
        # 特征分割索引
        self.split_indices = {
            'part1': slice(0, 9),
            'part2': slice(9, 17),
            'part3': slice(17, 25),
            'part4': slice(25, 33),
            'part5': slice(33, 41),
            'part6': slice(41, 50)
        }
        
    def _make_transform_layer(self, in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征分割和转换
        transformed_features = []
        for name, transform in self.feature_transforms.items():
            split_feature = x[:, self.split_indices[name]]
            transformed = transform(split_feature)
            transformed_features.append(transformed)
        
        # 特征堆叠
        stacked_features = torch.stack(transformed_features, dim=1)  # [B, 6, D]
        
        # 特征组注意力
        attended_features = self.group_attention(
            stacked_features.transpose(0, 1),
            stacked_features.transpose(0, 1),
            stacked_features.transpose(0, 1)
        )[0].transpose(0, 1)
        
        # 特征交互
        interacted_features = self.multi_scale_attention(attended_features)
        
        # 动态特征权重
        batch_size = x.size(0)
        flat_features = interacted_features.view(batch_size, -1)
        dynamic_weights = self.dynamic_weights(flat_features)
        weighted_features = (interacted_features * dynamic_weights.unsqueeze(-1)).sum(dim=1)
        
        # 特征增强
        enhanced_features = self.feature_enhancement(weighted_features)
        
        # 特征对比学习
        contrast_features = self.contrast_proj(enhanced_features)
        
        # 特征校准
        global_context = torch.cat([weighted_features, enhanced_features], dim=-1)
        calibration_weights = self.feature_calibration(global_context)
        calibrated_features = enhanced_features * calibration_weights
        
        # 最终特征聚合
        output_features = self.feature_aggregation(calibrated_features)
        output_features = self.output_norm(output_features)
        
        if self.training:
            return {
                'features': output_features,
                'contrast_features': contrast_features
            }
        else:
            return output_features

class WarmupCosineLRScheduler:
    """预热余弦退火学习率调度器
    
    参数:
        optimizer: 优化器实例
        warmup_epochs (int): 预热轮数
        max_epochs (int): 总训练轮数
        min_lr (float): 最小学习率，默认1e-6
        warmup_start_lr (float): 预热起始学习率，默认1e-7
        
    学习率变化:
    1. 预热阶段 [0, warmup_epochs]:
       lr = warmup_start_lr + (initial_lr - warmup_start_lr) * (current_epoch / warmup_epochs)
    2. 余弦退火阶段 [warmup_epochs, max_epochs]:
       lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * progress))
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self, epoch: int = None) -> float:
        if epoch is not None:
            self.current_epoch = epoch
            
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段
            lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # 余弦退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1
        return lr
        
    def state_dict(self):
        return {
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'min_lr': self.min_lr,
            'warmup_start_lr': self.warmup_start_lr,
            'initial_lr': self.initial_lr,
            'current_epoch': self.current_epoch
        }
        
    def load_state_dict(self, state_dict):
        self.warmup_epochs = state_dict['warmup_epochs']
        self.max_epochs = state_dict['max_epochs']
        self.min_lr = state_dict['min_lr']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.initial_lr = state_dict['initial_lr']
        self.current_epoch = state_dict['current_epoch']

class MAFN(nn.Module):
    """多尺度自适应融合网络 (Multi-scale Adaptive Fusion Network)
    
    输入维度: [batch_size, 50] - 原始特征向量
    输出维度: [batch_size, num_classes] - 类别预测概率
    
    模型架构:
    M (Multi-scale): 多尺度特征提取，使用三种尺度[1.0, 0.75, 0.5]提取特征
    A (Adaptive): 自适应特征融合，动态调整不同尺度特征的权重
    F (Fusion): 特征融合架构，包含通道注意力和空间注意力
    N (Network): 网络整体架构，包含特征金字塔和多尺度分类器
    
    参数:
        num_classes (int): 手势类别数
        feature_dim (int): 特征维度，默认128
        dropout_rate (float): Dropout比率，默认0.3
    """
    def __init__(self, num_classes: int, feature_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        
        self.feature_fusion = FeatureFusion(
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # 特征金字塔
        self.pyramid_scales = [1.0, 0.75, 0.5]
        self.pyramid_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, int(feature_dim * scale)),
                nn.LayerNorm(int(feature_dim * scale)),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for scale in self.pyramid_scales
        ])
        
        # 多尺度分类器
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(int(feature_dim * scale)),
                nn.Dropout(dropout_rate),
                nn.Linear(int(feature_dim * scale), num_classes)
            ) for scale in self.pyramid_scales
        ])
        
        # 特征蒸馏
        self.distill_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(feature_dim * scale), feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU()
            ) for scale in self.pyramid_scales
        ])
        
        # 分类器权重
        self.classifier_weights = nn.Parameter(torch.ones(len(self.pyramid_scales)))
        
        # 对比学习温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征融合
        fusion_output = self.feature_fusion(x)
        if isinstance(fusion_output, dict):
            fused_features = fusion_output['features']
            contrast_features = fusion_output['contrast_features']
        else:
            fused_features = fusion_output
            contrast_features = None
        
        # 特征金字塔
        pyramid_features = [
            transform(fused_features) 
            for transform in self.pyramid_transforms
        ]
        
        # 多尺度分类
        logits = []
        distill_features = []
        for features, classifier, distill in zip(pyramid_features, self.classifiers, self.distill_transforms):
            logits.append(classifier(features))
            distill_features.append(distill(features))
            
        # 加权融合预测结果
        weights = F.softmax(self.classifier_weights, dim=0)
        final_logits = sum(w * l for w, l in zip(weights, logits))
        
        if self.training and contrast_features is not None:
            return {
                'logits': final_logits,
                'aux_logits': logits,
                'contrast_features': contrast_features,
                'distill_features': distill_features
            }
        else:
            return final_logits

class ContrastiveLoss(nn.Module):
    """对比学习损失"""
    def __init__(self):
        super().__init__()
        
    def forward(self, features, labels, temperature):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        exp_logits = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()

class DistillationLoss(nn.Module):
    """知识蒸馏损失"""
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_features, teacher_features):
        """计算知识蒸馏损失
        
        Args:
            student_features: 学生模型特征 [B, D1]
            teacher_features: 教师模型特征 [B, D2]
            
        Returns:
            loss: 知识蒸馏损失
        """
        # 确保特征维度匹配
        if student_features.size(1) != teacher_features.size(1):
            # 使用线性投影将学生特征映射到与教师特征相同的维度
            projection = nn.Linear(
                student_features.size(1), 
                teacher_features.size(1),
                bias=False
            ).to(student_features.device)
            student_features = projection(student_features)
        
        # 计算概率分布
        student_probs = F.log_softmax(student_features / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_features / self.temperature, dim=1)
        
        # 计算KL散度
        loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        
        return loss

class MaskPredictor(nn.Module):
    """特征重建预测器
    
    用于自监督预训练阶段，预测被遮蔽的特征
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, x):
        return self.predictor(x)

class RotationPredictor(nn.Module):
    """旋转角度预测器
    
    用于自监督预训练阶段，预测特征旋转角度
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 4)  # 预测0°, 90°, 180°, 270°四种旋转
        )
    
    def forward(self, x):
        return self.predictor(x)

class PretrainMAFN(nn.Module):
    """预训练模型
    
    包含以下自监督任务：
    1. 特征重建：预测被遮蔽的特征
    2. 旋转预测：预测特征旋转角度
    3. 时序一致性：保持时序特征的一致性
    """
    def __init__(self, feature_dim=128, dropout_rate=0.3):
        super().__init__()
        
        # 特征提取器（与MAFN共享）
        self.feature_fusion = FeatureFusion(
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # 特征重建预测器
        self.mask_predictor = MaskPredictor(feature_dim)
        
        # 旋转预测器
        self.rotation_predictor = RotationPredictor(feature_dim)
        
        # 时序一致性投影头
        self.temporal_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 特征增强
        self.feature_augmentor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.feature_dim = feature_dim
        
    def _apply_rotation(self, features):
        """应用特征旋转
        
        Args:
            features: [batch_size, feature_dim] 的特征张量
            
        Returns:
            rot_features: [batch_size, feature_dim] 旋转后的特征
            rot_labels: [batch_size] 旋转角度标签
        """
        batch_size = features.size(0)
        device = features.device
        
        # 将特征重塑为方阵以便旋转
        side_len = int(math.sqrt(self.feature_dim))
        remainder = self.feature_dim - (side_len * side_len)
        
        rot_features = []
        rot_labels = []
        
        for i in range(batch_size):
            # 随机选择旋转角度
            rot_idx = torch.randint(0, 4, (1,)).item()
            
            # 将特征重塑为最接近的方形矩阵
            feat_square = features[i, :side_len*side_len].reshape(side_len, side_len)
            
            # 应用旋转
            rot_feat_square = torch.rot90(feat_square, k=rot_idx, dims=(0, 1))
            
            # 如果有剩余特征，保持不变
            if remainder > 0:
                rot_feat = torch.cat([
                    rot_feat_square.reshape(-1),
                    features[i, -remainder:]
                ])
            else:
                rot_feat = rot_feat_square.reshape(-1)
            
            rot_features.append(rot_feat)
            rot_labels.append(rot_idx)
        
        return torch.stack(rot_features), torch.tensor(rot_labels, device=device)
        
    def forward(self, x, mask_ratio=0.15):
        batch_size = x.size(0)
        
        # 生成遮蔽
        mask = torch.rand(batch_size, 50) > mask_ratio
        mask = mask.to(x.device)
        masked_x = x * mask.float()
        
        # 特征提取
        fusion_output = self.feature_fusion(masked_x)
        features = fusion_output['features'] if isinstance(fusion_output, dict) else fusion_output
        
        # 特征增强
        aug_features = self.feature_augmentor(features)
        
        # 应用特征旋转
        rot_features, rot_labels = self._apply_rotation(features)
        
        # 预测
        mask_pred = self.mask_predictor(features)
        rot_pred = self.rotation_predictor(rot_features)
        temp_proj = self.temporal_proj(features)
        
        return {
            'features': features,
            'aug_features': aug_features,
            'mask_pred': mask_pred,
            'rot_pred': rot_pred,
            'rot_labels': rot_labels,
            'temp_proj': temp_proj,
            'mask': mask
        }

class PretrainLoss(nn.Module):
    """预训练损失函数"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, outputs, original_features):
        # 特征重建损失 - 将预测特征投影回原始维度
        mask = outputs['mask']  # [B, 50]
        mask_pred = outputs['mask_pred']  # [B, feature_dim]
        
        # 使用线性投影将mask_pred投影回原始特征维度
        proj = nn.Linear(mask_pred.size(-1), original_features.size(-1)).to(mask_pred.device)
        mask_pred_proj = proj(mask_pred)
        
        recon_loss = F.mse_loss(
            mask_pred_proj * (~mask).float(),
            original_features * (~mask).float()
        )
        
        # 旋转预测损失
        rot_loss = F.cross_entropy(
            outputs['rot_pred'],
            outputs['rot_labels']
        )
        
        # 时序一致性损失（InfoNCE）
        features = F.normalize(outputs['features'], dim=1)
        aug_features = F.normalize(outputs['aug_features'], dim=1)
        temp_proj = F.normalize(outputs['temp_proj'], dim=1)
        
        sim_matrix = torch.matmul(temp_proj, aug_features.T) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        temp_loss = F.cross_entropy(sim_matrix, labels)
        
        # 总损失
        total_loss = recon_loss + rot_loss + temp_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'rot_loss': rot_loss,
            'temp_loss': temp_loss
        }

class AdaptiveFocalLoss(nn.Module):
    """自适应焦点损失
    
    动态调整alpha和gamma参数以处理类别不平衡
    """
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        
        # 如果没有提供alpha，则初始化为均匀权重
        if alpha is None:
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = alpha
            
        self.eps = 1e-6
        
        # 类别频率统计
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0))
        
    def update_class_weights(self, labels):
        """更新类别权重"""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        self.class_counts[unique_labels] += counts
        self.total_samples += len(labels)
        
        # 计算类别频率
        class_freq = self.class_counts / (self.total_samples + self.eps)
        
        # 使用有效样本数计算权重
        effective_num = 1.0 - torch.pow(0.999, self.class_counts)
        weights = (1.0 - class_freq) / (effective_num + self.eps)
        
        # 归一化权重
        self.alpha = weights / weights.sum()
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits [N, C]
            targets: 目标标签 [N]
        """
        # 更新类别权重
        self.update_class_weights(targets)
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        
        # 获取目标类别的alpha权重
        alpha = self.alpha[targets]
        
        # 动态调整gamma
        gamma = self.gamma * (1 - pt).detach()  # 难样本使用更大的gamma
        
        # 计算focal loss
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedHybridLoss(nn.Module):
    """改进的混合损失函数"""
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, smooth_factor=0.1):
        super().__init__()
        self.focal = AdaptiveFocalLoss(num_classes, alpha=alpha, gamma=gamma)
        self.label_smooth = LabelSmoothingCrossEntropy(smoothing=smooth_factor)
        self.contrast = ContrastiveLoss()
        self.distill = DistillationLoss()
        
        # 动态损失权重
        self.register_buffer('focal_weight', torch.tensor(1.0))
        self.register_buffer('smooth_weight', torch.tensor(0.1))
        self.register_buffer('contrast_weight', torch.tensor(0.1))
        self.register_buffer('distill_weight', torch.tensor(0.1))
        
        # 损失历史记录
        self.loss_history = {
            'focal': [],
            'smooth': [],
            'contrast': [],
            'distill': []
        }
        self.window_size = 10
        
        # 记录最新的损失值和权重
        self.latest_losses = {}
        self.latest_weights = {}
        
    def get_latest_stats(self):
        """获取最新的损失统计信息"""
        return {
            'losses': self.latest_losses,
            'weights': self.latest_weights
        }
        
    def update_weights(self, epoch, max_epochs, loss_values=None):
        """动态更新损失权重"""
        progress = epoch / max_epochs
        
        # 基于训练进度的基础权重调整
        self.focal_weight = 1.0 - 0.3 * progress
        self.smooth_weight = 0.1 + 0.1 * progress
        self.contrast_weight = 0.1 + 0.2 * progress
        self.distill_weight = 0.1 + 0.1 * progress
        
        # 如果提供了损失值，基于损失变化趋势进行微调
        if loss_values is not None:
            for loss_name, loss_val in loss_values.items():
                self.loss_history[loss_name].append(loss_val)
                if len(self.loss_history[loss_name]) > self.window_size:
                    self.loss_history[loss_name].pop(0)
            
            # 计算损失变化趋势
            trends = {}
            for loss_name, history in self.loss_history.items():
                if len(history) >= 2:
                    trend = (history[-1] - history[0]) / (len(history) - 1)
                    trends[loss_name] = trend
            
            # 基于趋势调整权重
            if trends:
                max_trend = max(abs(t) for t in trends.values())
                if max_trend > 0:
                    for loss_name, trend in trends.items():
                        if loss_name == 'focal':
                            self.focal_weight *= (1 + 0.1 * (trend / max_trend))
                        elif loss_name == 'smooth':
                            self.smooth_weight *= (1 + 0.1 * (trend / max_trend))
                        elif loss_name == 'contrast':
                            self.contrast_weight *= (1 + 0.1 * (trend / max_trend))
                        elif loss_name == 'distill':
                            self.distill_weight *= (1 + 0.1 * (trend / max_trend))
        
        # 归一化权重
        total_weight = (self.focal_weight + self.smooth_weight + 
                       self.contrast_weight + self.distill_weight)
        self.focal_weight /= total_weight
        self.smooth_weight /= total_weight
        self.contrast_weight /= total_weight
        self.distill_weight /= total_weight
        
    def forward(self, outputs, targets, epoch=None, max_epochs=None, teacher_outputs=None):
        """前向传播计算损失"""
        # 处理不同类型的输入
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
            outputs = {'logits': logits}
            
        # 计算各个损失
        focal_loss = self.focal(logits, targets)
        smooth_loss = self.label_smooth(logits, targets)
        
        if 'features' in outputs and outputs['features'] is not None:
            contrast_loss = self.contrast(outputs['features'], targets, self.temperature)
        else:
            contrast_loss = torch.tensor(0.0, device=logits.device)
            
        if teacher_outputs is not None and isinstance(teacher_outputs, dict) and 'features' in teacher_outputs:
            distill_loss = self.distill(outputs.get('features', logits), teacher_outputs['features'])
        else:
            distill_loss = torch.tensor(0.0, device=logits.device)
        
        # 更新权重
        if epoch is not None and max_epochs is not None:
            self.update_weights(
                epoch, 
                max_epochs,
                {
                    'focal': focal_loss.item(),
                    'smooth': smooth_loss.item(),
                    'contrast': contrast_loss.item(),
                    'distill': distill_loss.item()
                }
            )
            
        # 计算总损失
        total_loss = (
            self.focal_weight * focal_loss +
            self.smooth_weight * smooth_loss +
            self.contrast_weight * contrast_loss +
            self.distill_weight * distill_loss
        )
        
        # 保存最新的损失值和权重
        self.latest_losses = {
            'total_loss': total_loss.item(),
            'focal_loss': focal_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'distill_loss': distill_loss.item()
        }
        
        self.latest_weights = {
            'focal': self.focal_weight.item(),
            'smooth': self.smooth_weight.item(),
            'contrast': self.contrast_weight.item(),
            'distill': self.distill_weight.item()
        }
        
        # 直接返回总损失值
        return total_loss 
