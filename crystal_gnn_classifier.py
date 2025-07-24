#!/usr/bin/env python3
"""
晶体结构空间群分类的图神经网络实现

这个模块提供了一个完整的训练流程，用于使用GNN对晶体结构进行空间群分类。
包含数据预处理、模型构建、训练和评估等功能。

"""

import os
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 科学计算和机器学习
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as GeometricDataLoader

# 材料科学库
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.periodic_table import Element

# 数据处理和可视化
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class CrystalDataProcessor:
    """
    晶体数据预处理器
    
    负责从CIF文件中读取晶体结构，提取空间群标签，
    并将晶体结构转换为图数据格式。
    """
    
    def __init__(self, cif_dir: str, max_neighbors: int = 12, cutoff_radius: float = 5.0):
        """
        初始化数据处理器
        
        Args:
            cif_dir: CIF文件目录路径
            max_neighbors: 每个原子的最大邻居数
            cutoff_radius: 邻居搜索的截断半径(Å)
        """
        self.cif_dir = cif_dir
        self.max_neighbors = max_neighbors
        self.cutoff_radius = cutoff_radius
        self.voronoi_nn = VoronoiNN()
        
        # 原子特征维度
        self.atomic_features = {
            'atomic_number': 118,  # 原子序数 (1-118)
            'group': 18,           # 族
            'period': 7,           # 周期
            'electronegativity': 1, # 电负性
            'atomic_radius': 1,     # 原子半径
            'valence_electrons': 1  # 价电子数
        }
        
    def get_atomic_features(self, element: Element) -> np.ndarray:
        """
        提取原子特征
        
        Args:
            element: pymatgen Element对象
            
        Returns:
            原子特征向量
        """
        features = []
        
        # 原子序数 (one-hot编码)
        atomic_num_onehot = np.zeros(self.atomic_features['atomic_number'])
        atomic_num_onehot[element.Z - 1] = 1
        features.extend(atomic_num_onehot)
        
        # 族 (one-hot编码)
        group_onehot = np.zeros(self.atomic_features['group'])
        if element.group:
            group_onehot[element.group - 1] = 1
        features.extend(group_onehot)
        
        # 周期 (one-hot编码)
        period_onehot = np.zeros(self.atomic_features['period'])
        if element.row:
            period_onehot[element.row - 1] = 1
        features.extend(period_onehot)
        
        # 连续特征 (标准化)
        features.append(element.X if element.X else 0.0)  # 电负性
        features.append(element.atomic_radius if element.atomic_radius else 0.0)  # 原子半径
        
        # 价电子数 - 使用更安全的方式获取
        if hasattr(element, 'valence_electrons') and element.valence_electrons is not None:
            features.append(float(element.valence_electrons))
        else:
            # 如果没有valence_electrons属性，使用原子序数作为替代特征
            features.append(float(element.Z % 18))  # 使用原子序数模18作为近似价电子数
        
        return np.array(features, dtype=np.float32)
    
    def structure_to_graph(self, structure: Structure) -> Data:
        """
        将晶体结构转换为图数据
        
        Args:
            structure: pymatgen Structure对象
            
        Returns:
            PyTorch Geometric Data对象
        """
        # 获取原子特征
        node_features = []
        for site in structure:
            element = site.specie
            features = self.get_atomic_features(element)
            node_features.append(features)
        
        node_features = np.array(node_features)
        
        # 构建邻接关系
        edge_indices = []
        edge_features = []
        
        for i, site in enumerate(structure):
            try:
                # 使用Voronoi方法找邻居
                neighbors = self.voronoi_nn.get_nn_info(structure, i)
                
                for neighbor in neighbors[:self.max_neighbors]:  # 限制邻居数量
                    j = neighbor['site_index']
                    distance = neighbor['weight']  # Voronoi权重作为距离
                    
                    # 添加边 (双向)
                    edge_indices.extend([[i, j], [j, i]])
                    
                    # 边特征: 距离
                    edge_feature = [distance]
                    edge_features.extend([edge_feature, edge_feature])
                    
            except Exception as e:
                # 如果Voronoi失败，使用距离截断方法
                for j, other_site in enumerate(structure):
                    if i != j:
                        distance = structure.get_distance(i, j)
                        if distance <= self.cutoff_radius:
                            edge_indices.extend([[i, j], [j, i]])
                            edge_feature = [distance]
                            edge_features.extend([edge_feature, edge_feature])
        
        # 转换为tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # 如果没有边，创建自环
            num_nodes = len(structure)
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
            edge_attr = torch.zeros((num_nodes, 1), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def extract_space_group(self, cif_file: str) -> int:
        """
        从CIF文件中提取空间群编号
        
        Args:
            cif_file: CIF文件路径
            
        Returns:
            空间群编号 (1-230)
        """
        try:
            structure = Structure.from_file(cif_file)
            space_group = structure.get_space_group_info()[1]  # 获取空间群编号
            return space_group
        except Exception as e:
            # 如果无法从结构获取，尝试从文件名解析
            filename = os.path.basename(cif_file)
            if '_SG' in filename:
                try:
                    sg_part = filename.split('_SG')[1].split('.')[0]
                    return int(sg_part)
                except:
                    pass
            print(f"警告: 无法提取空间群信息 {cif_file}: {e}")
            return None
    
    def load_dataset(self, max_samples: Optional[int] = None) -> List[Tuple[Data, int]]:
        """
        加载整个数据集
        
        Args:
            max_samples: 最大样本数量，用于快速测试。None表示加载所有样本
        
        Returns:
            图数据和标签的列表
        """
        cif_files = glob.glob(os.path.join(self.cif_dir, '*.cif'))
        
        # 如果指定了最大样本数，则只处理前N个文件
        if max_samples is not None:
            cif_files = cif_files[:max_samples]
            print(f"快速测试模式: 只处理前 {len(cif_files)} 个文件")
        else:
            print(f"找到 {len(cif_files)} 个CIF文件")
        
        dataset = []
        failed_files = []
        
        for cif_file in tqdm(cif_files, desc="处理CIF文件"):
            try:
                # 加载结构
                structure = Structure.from_file(cif_file)
                
                # 提取空间群
                space_group = self.extract_space_group(cif_file)
                if space_group is None or space_group < 1 or space_group > 230:
                    failed_files.append(cif_file)
                    continue
                
                # 转换为图
                graph_data = self.structure_to_graph(structure)
                
                # 添加到数据集
                dataset.append((graph_data, space_group - 1))  # 转换为0-based索引
                
            except Exception as e:
                failed_files.append(cif_file)
                print(f"处理失败 {cif_file}: {e}")
        
        print(f"成功处理 {len(dataset)} 个文件")
        print(f"失败 {len(failed_files)} 个文件")
        
        return dataset

class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures
    """
    
    def __init__(self, data_list: List[Tuple[Data, int]]):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

class CrystalGNN(nn.Module):
    """
    图神经网络模型用于晶体结构分类
    
    使用GCN层进行图卷积，然后使用全局池化和全连接层进行分类。
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 num_layers: int = 3, 
                 num_classes: int = 230,
                 dropout: float = 0.2,
                 model_type: str = 'GCN'):
        """
        初始化GNN模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            num_classes: 分类类别数 (230个空间群)
            dropout: Dropout概率
            model_type: 模型类型 ('GCN' 或 'GIN')
        """
        super(CrystalGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        
        # 图卷积层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif model_type == 'GIN':
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 最后一层
        if num_layers > 1:
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 因为使用了mean和max池化
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 分类
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

class CrystalGNNTrainer:
    """
    GNN模型训练器
    
    负责模型训练、验证和评估的完整流程。
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        """
        初始化训练器
        
        Args:
            model: GNN模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=learning_rate, 
                                        weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.NLLLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: GeometricDataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="训练"):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: GeometricDataLoader) -> Tuple[float, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (验证损失, 验证准确率)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: GeometricDataLoader, 
              val_loader: GeometricDataLoader, 
              num_epochs: int = 100,
              save_path: str = 'best_model.pth') -> Dict:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
            
        Returns:
            训练历史字典
        """
        best_val_acc = 0
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_path)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  训练损失: {train_loss:.4f}")
                print(f"  验证损失: {val_loss:.4f}")
                print(f"  验证准确率: {val_acc:.4f}")
                print(f"  最佳验证准确率: {best_val_acc:.4f}")
                print(f"  当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def evaluate(self, test_loader: GeometricDataLoader) -> Dict:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试"):
                batch = batch.to(self.device)
                out = self.model(batch)
                pred = out.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'classification_report': classification_report(all_labels, all_preds),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }

def collate_fn(batch):
    """
    自定义批处理函数
    """
    data_list, labels = zip(*batch)
    batch_data = Batch.from_data_list(data_list)
    batch_data.y = torch.tensor(labels, dtype=torch.long)
    return batch_data

def plot_training_history(history: Dict, save_path: str = 'training_history.png'):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_path: 图片保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history['train_losses'], label='训练损失')
    ax1.plot(history['val_losses'], label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['val_accuracies'], label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.set_title('验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, save_path: str = 'confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 10))
    
    # 由于空间群太多，只显示对角线附近的区域
    sns.heatmap(cm, cmap='Blues', fmt='d', cbar=True)
    plt.title('空间群分类混淆矩阵')
    plt.xlabel('预测空间群')
    plt.ylabel('真实空间群')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 这里是主函数，将在下一个文件中实现
    pass