# 信息论与机器学习：交叉熵在肌电信号识别中的应用研究

## 项目简介

本项目基于信息论原理，研究交叉熵损失函数在肌电信号模式识别任务中的应用。通过对比交叉熵损失与均方误差损失在CNN模型上的表现，验证交叉熵在分类任务中的理论优越性和实践价值。

## 项目结构

```
项目根目录/
├── main.py                    # 主程序入口
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包列表
├── data/                      # 数据目录
│   └── Samples.npz            # 肌电信号数据文件
├── models/                    # 模型定义
│   └── cnn_model.py           # CNN模型架构
├── core/                      # 核心训练逻辑
│   └── trainer.py             # 训练流程控制
├── utils/                     # 工具函数
│   ├── data_loader.py         # 数据加载与预处理
│   ├── data_preprocessor.py   # 数据预处理
│   ├── callbacks.py           # 自定义回调函数
│   ├── visualization.py       # 可视化工具
│   └── performance_report.py  # 性能报告生成
├── results/                   # 输出结果
│   ├── best_model.keras       # 最佳模型
│   ├── classifier_model.keras # 分类器模型
│   ├── translator.json        # 标签映射文件
│   ├── model_comparison.png   # 模型对比图
│   └── loss_curve.png         # 损失曲线图
└── docs/                      # 文档与图片
    └── images/                # 项目相关图片
```

## 主要功能模块

1. **数据加载与预处理** (`utils/data_loader.py`, `utils/data_preprocessor.py`)
   - 加载肌电信号数据
   - 数据重构与窗口滑动处理
   - 标签编码与数据划分

2. **模型定义** (`models/cnn_model.py`)
   - 实现专用的CNN架构用于肌电信号分类
   - 包含深度可分离卷积等优化结构

3. **训练流程** (`core/trainer.py`)
   - 完整的模型训练、验证、评估流程
   - 支持交叉熵与均方误差损失对比实验
   - 模型保存与性能报告生成

4. **可视化工具** (`utils/visualization.py`)
   - 模型性能对比图表
   - 训练过程监控

5. **性能分析** (`utils/performance_report.py`)
   - 生成详细的训练性能报告
   - 收敛稳定性分析

## 使用方法

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 运行主程序

```bash
python main.py
```

### 自定义配置

在 `main.py` 中可以修改以下参数：
- `save_path`: 结果保存路径
- `samples_file`: 数据文件路径
- `sample_length`: 样本长度
- `model_type`: 使用的模型类型

## 实验结果

实验结果表明，在肌电信号模式识别任务中：
- 交叉熵损失相比均方误差损失，训练效率提升45%
- 识别准确率提高8%（90% vs 82%）
- 收敛稳定性显著改善

## 依赖包

```
tensorflow>=2.0.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

详细依赖见 `requirements.txt` 文件。

## 贡献者

- 【NAME】 (【ID】)

## 许可证

本项目仅供学术研究使用。