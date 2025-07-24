# EasyCrystal
一个处理晶体学数据的代码。

作者保留一切权利。

# 分支：yyx更新部分

## 数据集
cif_data存有5000份从materials project下载的cif数据，供训练
也可调整download_dataset.py脚本下载指定数据

## 图神经网络分类部分
运行train_crystal_gnn.py即可开始训练GNN模型，完成从cif文件中提取数据->读取空间群特征结构->构造训练集和验证集->训练模型的一系列步骤

## 其他
其他功能及其整合、模型调优、bug修复进行中
