#!/usr/bin/env python3
"""
晶体结构空间群分类训练脚本

这个脚本实现了完整的训练流程，包括数据加载、模型训练、验证和测试。

使用方法:
    python train_crystal_gnn.py

"""

import os
import argparse
import json
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from crystal_gnn_classifier import (
    CrystalDataProcessor,
    CrystalDataset,
    CrystalGNN,
    CrystalGNNTrainer,
    collate_fn,
    plot_training_history,
    plot_confusion_matrix
)

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='晶体结构空间群分类训练')
    
    # 数据参数
    parser.add_argument('--cif_dir', type=str, default='cif_data',
                       help='CIF文件目录路径')
    parser.add_argument('--max_neighbors', type=int, default=12,
                       help='每个原子的最大邻居数')
    parser.add_argument('--cutoff_radius', type=float, default=5.0,
                       help='邻居搜索截断半径(Å)')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GIN'],
                       help='GNN模型类型')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout概率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    
    # 数据划分参数
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """
    设置计算设备
    
    Args:
        device_arg: 设备参数
        
    Returns:
        PyTorch设备对象
    """
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def setup_directories(save_dir: str) -> str:
    """
    设置保存目录
    
    Args:
        save_dir: 基础保存目录
        
    Returns:
        时间戳目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(save_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    
    return run_dir

def save_config(args, run_dir: str):
    """
    保存配置参数
    
    Args:
        args: 命令行参数
        run_dir: 运行目录
    """
    config = vars(args)
    config_path = os.path.join(run_dir, 'config.json')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {config_path}")

def analyze_dataset(dataset, run_dir: str):
    """
    分析数据集统计信息
    
    Args:
        dataset: 数据集
        run_dir: 运行目录
    """
    print("\n=== 数据集分析 ===")
    
    # 统计空间群分布
    space_groups = [label for _, label in dataset]
    unique_sgs, counts = np.unique(space_groups, return_counts=True)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"空间群种类: {len(unique_sgs)}")
    print(f"最常见空间群: {unique_sgs[np.argmax(counts)] + 1} (出现 {np.max(counts)} 次)")
    print(f"最少见空间群: {unique_sgs[np.argmin(counts)] + 1} (出现 {np.min(counts)} 次)")
    
    # 绘制空间群分布
    plt.figure(figsize=(12, 6))
    plt.bar(unique_sgs + 1, counts)  # +1 转换回1-based索引
    plt.xlabel('空间群编号')
    plt.ylabel('样本数量')
    plt.title('空间群分布')
    plt.yscale('log')  # 使用对数刻度
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(run_dir, 'plots', 'space_group_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计图结构信息
    num_nodes = [data.x.size(0) for data, _ in dataset]
    num_edges = [data.edge_index.size(1) for data, _ in dataset]
    
    print(f"\n节点数统计:")
    print(f"  平均: {np.mean(num_nodes):.1f}")
    print(f"  中位数: {np.median(num_nodes):.1f}")
    print(f"  范围: {np.min(num_nodes)} - {np.max(num_nodes)}")
    
    print(f"\n边数统计:")
    print(f"  平均: {np.mean(num_edges):.1f}")
    print(f"  中位数: {np.median(num_edges):.1f}")
    print(f"  范围: {np.min(num_edges)} - {np.max(num_edges)}")
    
    # 保存统计信息
    stats = {
        'dataset_size': len(dataset),
        'num_space_groups': len(unique_sgs),
        'space_group_distribution': {int(sg + 1): int(count) for sg, count in zip(unique_sgs, counts)},
        'node_stats': {
            'mean': float(np.mean(num_nodes)),
            'median': float(np.median(num_nodes)),
            'min': int(np.min(num_nodes)),
            'max': int(np.max(num_nodes))
        },
        'edge_stats': {
            'mean': float(np.mean(num_edges)),
            'median': float(np.median(num_edges)),
            'min': int(np.min(num_edges)),
            'max': int(np.max(num_edges))
        }
    }
    
    stats_path = os.path.join(run_dir, 'dataset_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def split_dataset(dataset, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """
    划分数据集
    
    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        (训练集, 验证集, 测试集)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 提取标签用于分层抽样
    labels = [label for _, label in dataset]
    indices = list(range(len(dataset)))
    
    # 首先分离出测试集
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=seed, 
        stratify=labels
    )
    
    # 然后从训练+验证集中分离出验证集
    train_val_labels = [labels[i] for i in train_val_indices]
    val_size = val_ratio / (train_ratio + val_ratio)
    
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=seed,
        stratify=train_val_labels
    )
    
    # 创建子数据集
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    print(f"\n=== 数据集划分 ===")
    print(f"训练集: {len(train_dataset)} 样本 ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"验证集: {len(val_dataset)} 样本 ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"测试集: {len(test_dataset)} 样本 ({len(test_dataset)/len(dataset)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int):
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        batch_size: 批大小
        num_workers: 工作进程数
        
    Returns:
        (训练加载器, 验证加载器, 测试加载器)
    """
    train_loader = GeometricDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    val_loader = GeometricDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    test_loader = GeometricDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备和目录
    device = setup_device(args.device)
    run_dir = setup_directories(args.save_dir)
    save_config(args, run_dir)
    
    print("\n=== 晶体结构空间群分类训练 ===")
    print(f"运行目录: {run_dir}")
    
    # 1. 数据预处理
    print("\n=== 步骤1: 数据预处理 ===")
    processor = CrystalDataProcessor(
        cif_dir=args.cif_dir,
        max_neighbors=args.max_neighbors,
        cutoff_radius=args.cutoff_radius
    )
    
    # 加载数据集
    print("加载数据集...")
    dataset = processor.load_dataset()
    
    if len(dataset) == 0:
        print("错误: 没有成功加载任何数据")
        return
    
    # 分析数据集
    analyze_dataset(dataset, run_dir)
    
    # 2. 数据集划分
    print("\n=== 步骤2: 数据集划分 ===")
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, args.batch_size, args.num_workers
    )
    
    # 3. 模型构建
    print("\n=== 步骤3: 模型构建 ===")
    
    # 计算输入特征维度
    sample_data, _ = dataset[0]
    input_dim = sample_data.x.size(1)
    
    print(f"输入特征维度: {input_dim}")
    print(f"输出类别数: 230 (空间群)")
    
    model = CrystalGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=230,
        dropout=args.dropout,
        model_type=args.model_type
    )
    
    print(f"模型类型: {args.model_type}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 模型训练
    print("\n=== 步骤4: 模型训练 ===")
    
    trainer = CrystalGNNTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 训练模型
    model_save_path = os.path.join(run_dir, 'models', 'best_model.pth')
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_path=model_save_path
    )
    
    print(f"\n训练完成！最佳验证准确率: {history['best_val_acc']:.4f}")
    
    # 保存训练历史
    history_path = os.path.join(run_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_path = os.path.join(run_dir, 'plots', 'training_history.png')
    plot_training_history(history, plot_path)
    
    # 5. 模型评估
    print("\n=== 步骤5: 模型评估 ===")
    
    # 加载最佳模型
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    results = trainer.evaluate(test_loader)
    
    print(f"\n测试集结果:")
    print(f"准确率: {results['accuracy']:.4f}")
    
    # 保存评估结果
    results_path = os.path.join(run_dir, 'test_results.json')
    results_to_save = {
        'accuracy': results['accuracy'],
        'classification_report': results['classification_report']
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(run_dir, 'plots', 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], cm_path)
    
    # 保存详细的分类报告
    report_path = os.path.join(run_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("晶体结构空间群分类结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试集准确率: {results['accuracy']:.4f}\n\n")
        f.write("详细分类报告:\n")
        f.write(results['classification_report'])
    
    print(f"\n=== 训练完成 ===")
    print(f"所有结果已保存到: {run_dir}")
    print(f"最佳模型: {model_save_path}")
    print(f"测试准确率: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()