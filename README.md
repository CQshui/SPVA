# 数字全息自动聚焦 (Local-Structure Autofocusing)

利用边缘检测网络提取重建图像的局部结构特征，结合贝叶斯优化自动搜索最优聚焦距离。

## 方法流程

原始全息图 → 频谱滤波 → 角谱法重建(z) → 边缘检测(RCF) → 聚焦分数 → 贝叶斯优化 → 最优z

## 核心模块

| 模块 | 功能 |
|------|------|
| propagate/ | 角谱法传播，GPU加速的全息图数值重建 |
| simulation/ | 生成仿真全息图（同轴/离轴/含散斑噪声） |
| edge_prediction/ | 边缘检测：多种网络模型(RCF/BDCN/DexiNed等)、训练、预测 |
| peak_finder/ | 贝叶斯优化寻峰，基于代理模型在z区间内搜索最优值 |
| comparison_method/ | 对比方法：方差法、GoG/ToG、特征值法、FocusNet |
| test/ | COCO数据集和实验数据上的测试脚本 |
| plot_data/ | 结果可视化与误差分析 |

## 快速开始

### 1. 生成仿真数据集

修改 simulation/simulation.py 中的输入输出路径及相关参数（波长、像素大小、传播类型等），然后运行：

python simulation/simulation.py

### 2. 训练边缘检测模型

修改 edge_prediction/train/train.py 中的数据集路径、batch_size、maxepoch、learning_rate 等参数，然后运行：

python edge_prediction/train/train.py

### 3. 预测与测试

修改 test/COCO/test_singlePlane_COCO.py 中的全息图路径、模型权重路径、z值搜索区间(lb/ub)等参数，然后运行：

python test/COCO/test_singlePlane_COCO.py

## 环境依赖

Python 3.8+, PyTorch, CUDA, OpenCV, NumPy, SciPy, Matplotlib, Pillow, tqdm, natsort

## 项目结构

```
propagate/angular_torch.py         角谱法传播
simulation/simulation.py           基础全息图仿真
simulation/simulation_speckle.py   含散斑噪声仿真
simulation/simulation_gray_compress.py  灰度压缩仿真
edge_prediction/model/             RCF, BDCN, DexiNed等模型
edge_prediction/train/train.py     训练
edge_prediction/dataloader/        数据加载
edge_prediction/predict.py         预测与聚焦分数
peak_finder/peakfinder.py          贝叶斯优化
comparison_method/global_methods/  variance, GoG/ToG, eig
comparison_method/FocusNet/        深度学习方法
test/COCO/                         COCO测试
test/Experiment/                   实验数据测试
plot_data/                         可视化分析
```

## 参考文献

- RCF: Richer Convolutional Features for Edge Detection, CVPR 2017
- FocusNet: Learning Autofocus from a Few Examples, 2024
