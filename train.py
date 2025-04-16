import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from model import (
    MAFN, WarmupCosineLRScheduler,
    ContrastiveLoss, DistillationLoss, PretrainMAFN, PretrainLoss,
    ImprovedHybridLoss
)
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import copy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GestureDataset(Dataset):
    """手势数据集加载器，包含数据增强"""
    def __init__(self, features, labels, is_training=True):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.is_training = is_training
        
    def __len__(self):
        return len(self.features)
    
    def add_noise(self, x):
        """添加高斯噪声，降低噪声强度"""
        noise = torch.randn_like(x) * 0.005  # 降低噪声强度
        return x + noise
    
    def feature_mask(self, x):
        """随机特征遮蔽，降低遮蔽概率"""
        mask = torch.rand_like(x) > 0.02  # 降低遮蔽概率到2%
        return x * mask
    
    def feature_jitter(self, x):
        """特征抖动，使用更小的抖动幅度"""
        return x + torch.randn_like(x) * x.std(dim=0) * 0.005
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.is_training:
            # 降低数据增强的概率
            if torch.rand(1) < 0.3:  # 降低到30%的概率
                x = self.add_noise(x)
            if torch.rand(1) < 0.2:  # 降低到20%的概率
                x = self.feature_mask(x)
            if torch.rand(1) < 0.2:  # 降低到20%的概率
                x = self.feature_jitter(x)
                
        return x, y

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, is_english=False):
    """绘制混淆矩阵（百分比）"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix (%)' if is_english else '混淆矩阵 (%)')
    plt.xlabel('Predicted Label' if is_english else '预测标签')
    plt.ylabel('True Label' if is_english else '真实标签')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(metrics, save_path, is_pretrain=True):
    """绘制训练曲线
    
    Args:
        metrics (dict): 包含训练指标的字典
        save_path (str): 保存路径
        is_pretrain (bool): 是否是预训练阶段
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='训练损失', linewidth=2)
    plt.plot(metrics['epoch'], metrics['val_loss'], label='验证损失', linewidth=2)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.title('训练和验证损失曲线', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 绘制学习率曲线
    plt.subplot(2, 1, 2)
    plt.plot(metrics['epoch'], metrics['lr'], label='学习率', linewidth=2, color='green')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.title('学习率变化曲线', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 保存中文版本
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves_zh.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建英文版本
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线（英文）
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 绘制学习率曲线（英文）
    plt.subplot(2, 1, 2)
    plt.plot(metrics['epoch'], metrics['lr'], label='Learning Rate', linewidth=2, color='green')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 保存英文版本
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves_en.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 如果是微调阶段，额外绘制准确率曲线
    if not is_pretrain and 'train_acc' in metrics and 'val_acc' in metrics:
        # 中文版本
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['train_acc'], label='训练准确率', linewidth=2)
        plt.plot(metrics['epoch'], metrics['val_acc'], label='验证准确率', linewidth=2)
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.title('训练和验证准确率曲线', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'accuracy_curves_zh.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 英文版本
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['train_acc'], label='Training Accuracy', linewidth=2)
        plt.plot(metrics['epoch'], metrics['val_acc'], label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training and Validation Accuracy Curves', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'accuracy_curves_en.png'), dpi=300, bbox_inches='tight')
        plt.close()

def train_with_mixup(x, y, alpha=0.1):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    save_dir,
    device,
    is_pretrain=True,
    patience=10,
    class_names=None,
    class_names_en=None
):
    """训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        save_dir: 模型保存目录
        device: 训练设备
        is_pretrain: 是否是预训练阶段
        patience: 早停耐心值
        class_names: 类别名称列表
        class_names_en: 英文类别名称列表
        
    Returns:
        model: 训练好的模型
    """
    os.makedirs('results', exist_ok=True)
    
    best_metric = float('inf') if is_pretrain else 0
    best_epoch = 0
    counter = 0
    
    metrics = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []
    }
    
    if not is_pretrain:
        metrics.update({
            'train_acc': [],
            'val_acc': []
        })
    
    # 梯度累积步数
    accumulation_steps = 2
    
    # 仅在微调时使用
    if not is_pretrain:
        # EMA模型
        ema_model = copy.deepcopy(model)
        ema_decay = 0.999
        
        # 损失函数
        contrast_criterion = ContrastiveLoss().to(device)
        distill_criterion = DistillationLoss().to(device)
        
        # 损失权重（动态调整）
        loss_weights = {
            'main': 1.0,
            'aux': 0.3,
            'contrast': 0.1,
            'distill': 0.1
        }
    
    for epoch in range(num_epochs):
        model.train()
        if not is_pretrain:
            ema_model.eval()
        
        train_stats = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            is_pretrain=is_pretrain,
            ema_model=ema_model if not is_pretrain else None,
            loss_weights=loss_weights if not is_pretrain else None,
            contrast_criterion=contrast_criterion if not is_pretrain else None,
            distill_criterion=distill_criterion if not is_pretrain else None,
            epoch=epoch,
            num_epochs=num_epochs
        )
        
        # 更新学习率
        current_lr = scheduler.step()
        
        # 验证
        val_stats = validate_epoch(
            model=ema_model if not is_pretrain else model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            is_pretrain=is_pretrain
        )
        
        # 更新和记录指标
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_stats['loss'])
        metrics['val_loss'].append(val_stats['loss'])
        metrics['lr'].append(current_lr)
        
        if not is_pretrain:
            metrics['train_acc'].append(train_stats['acc'])
            metrics['val_acc'].append(val_stats['acc'])
        
        # 保存指标
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(
            os.path.join('results', 'pretrain_metrics.csv' if is_pretrain else 'training_metrics.csv'),
            index=False
        )
        
        # 打印当前轮次信息
        print(f'\n{"预训练" if is_pretrain else "训练"}轮次 {epoch+1}/{num_epochs}:')
        print(f'学习率: {current_lr:.6f}')
        print(f'训练损失: {train_stats["loss"]:.4f}')
        print(f'验证损失: {val_stats["loss"]:.4f}')
        
        if not is_pretrain:
            print(f'训练准确率: {train_stats["acc"]:.2f}%')
            print(f'验证准确率: {val_stats["acc"]:.2f}%')
            
            # 定期保存混淆矩阵
            if (epoch + 1) in [num_epochs // 4, num_epochs // 2, num_epochs * 3 // 4, num_epochs]:
                if class_names is not None:
                    plot_confusion_matrix(
                        val_stats['labels'],
                        val_stats['preds'],
                        class_names,
                        f'results/confusion_matrix_epoch_{epoch+1}.pdf',
                        is_english=False
                    )
                    
                    report = classification_report(
                        val_stats['labels'],
                        val_stats['preds'],
                        target_names=class_names,
                        output_dict=True
                    )
                    pd.DataFrame(report).transpose().to_csv(
                        f'results/classification_report_epoch_{epoch+1}.csv'
                    )
        
        # 保存最佳模型
        current_metric = val_stats['loss'] if is_pretrain else val_stats['acc']
        is_best = current_metric < best_metric if is_pretrain else current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            
            if is_pretrain:
                save_dict['best_val_loss'] = best_metric
            else:
                save_dict['best_val_acc'] = best_metric
                save_dict['train_metrics'] = metrics
                # 保存EMA模型
                save_dict['ema_model_state_dict'] = ema_model.state_dict()
            
            torch.save(
                save_dict,
                'results/best_pretrain_model.pth' if is_pretrain else 'results/best_model.pth'
            )
            
            if not is_pretrain:
                # 保存中文混淆矩阵
                if class_names is not None:
                    plot_confusion_matrix(
                        val_stats['labels'],
                        val_stats['preds'],
                        class_names,
                        'results/best_confusion_matrix_zh.pdf',
                        is_english=False
                    )
                    
                    # 保存英文混淆矩阵
                    plot_confusion_matrix(
                        val_stats['labels'],
                        val_stats['preds'],
                        class_names_en,
                        'results/best_confusion_matrix_en.pdf',
                        is_english=True
                    )
                
                print(f'保存最佳{"预训练" if is_pretrain else ""}模型，'
                      f'验证{"损失" if is_pretrain else "准确率"}: {best_metric:.4f}')
            counter = 0
        else:
            counter += 1
        
        # 早停
        if counter >= patience:
            print(f'验证{"损失" if is_pretrain else "准确率"}已连续{patience}轮未{"下降" if is_pretrain else "提升"}，停止训练')
            if not is_pretrain:
                print(f'最佳验证准确率: {best_metric:.2f}% (轮次 {best_epoch+1})')
            break
    
    # 绘制训练曲线
    plot_training_curves(metrics, save_dir, is_pretrain)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'results/best_pretrain_model.pth' if is_pretrain else 'results/best_model.pth')['model_state_dict'])
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps,
                is_pretrain, ema_model=None, loss_weights=None, contrast_criterion=None,
                distill_criterion=None, epoch=0, num_epochs=0):
    """训练一个轮次"""
    model.train()
    
    train_loss = 0
    train_correct = 0 if not is_pretrain else None
    train_total = 0 if not is_pretrain else None
    all_train_preds = [] if not is_pretrain else None
    all_train_labels = [] if not is_pretrain else None
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'{"预训练" if is_pretrain else "训练"}轮次 {epoch+1}/{num_epochs}')
    for batch_idx, (features, labels) in enumerate(pbar):
        features, labels = features.to(device), labels.to(device)
        
        if is_pretrain:
            # 预训练前向传播
            outputs = model(features)
            losses = criterion(outputs, features)
            loss = losses['total_loss']
        else:
            # 微调前向传播
            mixed_features, labels_a, labels_b, lam = train_with_mixup(features, labels)
            outputs = model(mixed_features)
            
            # 计算各种损失
            main_loss = lam * criterion(outputs['logits'], labels_a) + \
                       (1 - lam) * criterion(outputs['logits'], labels_b)
            main_loss = main_loss * loss_weights['main']
            
            aux_losses = []
            for aux_out in outputs['aux_logits']:
                aux_loss = lam * criterion(aux_out, labels_a) + \
                          (1 - lam) * criterion(aux_out, labels_b)
                aux_losses.append(aux_loss)
            
            contrast_loss = contrast_criterion(
                outputs['contrast_features'], labels, model.temperature
            ) * loss_weights['contrast']
            
            distill_loss = sum(
                distill_criterion(student_feat, outputs['contrast_features'].detach())
                for student_feat in outputs['distill_features']
            ) * loss_weights['distill'] / len(outputs['distill_features'])
            
            loss = main_loss + sum(aux_losses) * loss_weights['aux'] + \
                   contrast_loss + distill_loss
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if not is_pretrain:
                # 更新EMA模型
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(0.999).add_(param.data, alpha=1 - 0.999)
        
        train_loss += loss.item() * accumulation_steps
        
        if not is_pretrain:
            _, predicted = outputs['logits'].max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        if is_pretrain:
            pbar.set_postfix({
                '损失': f'{loss.item() * accumulation_steps:.4f}'
            })
        else:
            pbar.set_postfix({
                '损失': f'{loss.item() * accumulation_steps:.4f}',
                '准确率': f'{100.*train_correct/train_total:.2f}%'
            })
    
    # 计算平均损失
    avg_loss = train_loss / len(train_loader)
    
    if is_pretrain:
        return {'loss': avg_loss}
    else:
        return {
            'loss': avg_loss,
            'acc': 100. * train_correct / train_total,
            'preds': all_train_preds,
            'labels': all_train_labels
        }

def validate_epoch(model, val_loader, criterion, device, is_pretrain):
    """验证一个轮次"""
    model.eval()
    
    val_loss = 0
    val_correct = 0 if not is_pretrain else None
    val_total = 0 if not is_pretrain else None
    all_val_preds = [] if not is_pretrain else None
    all_val_labels = [] if not is_pretrain else None
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            if is_pretrain:
                # 预训练验证
                outputs = model(features)
                losses = criterion(outputs, features)
                val_loss += losses['total_loss'].item()
            else:
                # 微调验证
                outputs = model(features)
                # 处理不同类型的输出
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                    
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失
    avg_loss = val_loss / len(val_loader)
    
    if is_pretrain:
        return {'loss': avg_loss}
    else:
        return {
            'loss': avg_loss,
            'acc': 100. * val_correct / val_total,
            'preds': all_val_preds,
            'labels': all_val_labels
        }

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    features = np.load('processed_data/features.npy')
    labels = np.load('processed_data/labels.npy')
    labels_en = np.load('processed_data/labels_en.npy')
    
    # 获取类别名称
    class_names = np.unique(labels)
    class_names_en = np.unique(labels_en)
    
    # 转换标签为数值
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    numeric_labels = np.array([label_to_idx[label] for label in labels])
    
    # 划分数据集
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
    )
    
    # 创建数据加载器
    train_dataset = GestureDataset(train_features, train_labels, is_training=True)
    val_dataset = GestureDataset(val_features, val_labels, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 设置默认的CUDA设备
        torch.cuda.set_device(0)
        # 启用CUDA性能优化
        torch.backends.cudnn.benchmark = True
        print(f'使用GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('警告：未检测到GPU，将使用CPU进行训练（这可能会很慢）')
    
    # 创建预训练模型
    pretrain_model = PretrainMAFN(feature_dim=128, dropout_rate=0.3).to(device)
    
    # 预训练优化器和调度器
    pretrain_optimizer = optim.AdamW(
        pretrain_model.parameters(),
        lr=0.0001,
        weight_decay=0.05
    )
    
    pretrain_scheduler = WarmupCosineLRScheduler(
        pretrain_optimizer,
        warmup_epochs=10,
        max_epochs=100,
        min_lr=1e-6,
        warmup_start_lr=1e-7
    )
    
    # 预训练
    pretrain_criterion = PretrainLoss().to(device)
    pretrain_model = train_model(
        model=pretrain_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=pretrain_criterion,
        optimizer=pretrain_optimizer,
        scheduler=pretrain_scheduler,
        num_epochs=100,
        save_dir='results',
        device=device,
        is_pretrain=True,
        class_names=None,
        class_names_en=None
    )
    
    # 创建微调模型
    model = MAFN(
        num_classes=len(class_names),
        feature_dim=128,
        dropout_rate=0.3
    ).to(device)
    
    # 加载预训练权重
    model.feature_fusion.load_state_dict(
        pretrain_model.feature_fusion.state_dict()
    )
    
    # 微调优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = WarmupCosineLRScheduler(
        optimizer,
        warmup_epochs=10,
        max_epochs=250,
        min_lr=1e-6,
        warmup_start_lr=1e-7
    )
    
    # 使用改进的混合损失函数
    criterion = ImprovedHybridLoss(
        num_classes=len(class_names),
        alpha=0.25,
        gamma=2.0,
        smooth_factor=0.1
    ).to(device)
    
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=250,
        save_dir='results',
        device=device,
        is_pretrain=False,
        class_names=class_names,
        class_names_en=class_names_en
    )

if __name__ == '__main__':
    main() 
