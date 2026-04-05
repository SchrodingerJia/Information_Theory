import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(history_ce, history_mse, model_names=('交叉熵模型', '均方误差模型')):
    """
    绘制两个模型的对比曲线：损失和准确率
    参数:
    history_ce -- 使用交叉熵损失的模型训练历史
    history_mse -- 使用均方误差损失的模型训练历史
    model_names -- 两个模型的名称元组
    """
    # 创建2x2的子图布局
    plt.figure(figsize=(18, 12))
    plt.suptitle('模型性能对比分析', fontsize=18, fontweight='bold')
    # ==================== 1. 训练损失对比 ====================
    plt.subplot(2, 2, 1)
    plt.plot(history_ce.history['loss'], 'b-', linewidth=2, label=model_names[0])
    plt.plot(history_mse.history['loss'], 'r-', linewidth=2, label=model_names[1])
    plt.title('训练损失对比', fontsize=15)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    # ==================== 2. 验证损失对比 ====================
    plt.subplot(2, 2, 2)
    # 检查是否有验证损失数据
    if 'val_loss' in history_ce.history and 'val_loss' in history_mse.history:
        plt.plot(history_ce.history['val_loss'], 'b-', linewidth=2, label=model_names[0])
        plt.plot(history_mse.history['val_loss'], 'r-', linewidth=2, label=model_names[1])
        plt.title('验证损失对比', fontsize=15)
    else:
        plt.text(0.5, 0.5, '无验证损失数据',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=14)
        plt.title('验证损失对比', fontsize=15)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    # ==================== 3. 训练准确率对比 ====================
    plt.subplot(2, 2, 3)
    # 检查历史记录中使用的准确率键名
    ce_acc_key = 'accuracy' if 'accuracy' in history_ce.history else 'acc'
    mse_acc_key = 'accuracy' if 'accuracy' in history_mse.history else 'acc'
    plt.plot(history_ce.history[ce_acc_key], 'b-', linewidth=2, label=model_names[0])
    plt.plot(history_mse.history[mse_acc_key], 'r-', linewidth=2, label=model_names[1])
    plt.title('训练准确率对比', fontsize=15)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0, 1.05])  # 固定y轴范围
    plt.legend(fontsize=12)
    # ==================== 4. 验证准确率对比 ====================
    plt.subplot(2, 2, 4)
    # 检查是否有验证准确率数据
    ce_val_key = 'val_accuracy' if 'val_accuracy' in history_ce.history else 'val_acc'
    mse_val_key = 'val_accuracy' if 'val_accuracy' in history_mse.history else 'val_acc'
    if ce_val_key in history_ce.history and mse_val_key in history_mse.history:
        plt.plot(history_ce.history[ce_val_key], 'b-', linewidth=2, label=model_names[0])
        plt.plot(history_mse.history[mse_val_key], 'r-', linewidth=2, label=model_names[1])
        plt.title('验证准确率对比', fontsize=15)
    else:
        plt.text(0.5, 0.5, '无验证准确率数据',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=14)
        plt.title('验证准确率对比', fontsize=15)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0, 1.05])
    plt.legend(fontsize=12)
    # 添加最终性能指标
    final_ce_acc = history_ce.history[ce_acc_key][-1]
    final_mse_acc = history_mse.history[mse_acc_key][-1]
    if ce_val_key in history_ce.history and mse_val_key in history_mse.history:
        final_ce_val_acc = history_ce.history[ce_val_key][-1]
        final_mse_val_acc = history_mse.history[mse_val_key][-1]
        plt.figtext(0.5, 0.01,
                   f"{model_names[0]} 最终训练准确率: {final_ce_acc:.4f} | 最终验证准确率: {final_ce_val_acc:.4f}\n"
                   f"{model_names[1]} 最终训练准确率: {final_mse_acc:.4f} | 最终验证准确率: {final_mse_val_acc:.4f}",
                   ha="center", fontsize=13, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    else:
        plt.figtext(0.5, 0.01,
                   f"{model_names[0]} 最终训练准确率: {final_ce_acc:.4f}\n"
                   f"{model_names[1]} 最终训练准确率: {final_mse_acc:.4f}",
                   ha="center", fontsize=13, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 为底部文本留出空间
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_stability(loss_values, window=5):
    """计算最后N个epoch的损失方差"""
    last_n_losses = loss_values[-window:]
    return np.var(last_n_losses)