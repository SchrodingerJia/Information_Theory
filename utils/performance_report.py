def generate_performance_report(history, timer):
    """生成完整性能报告"""
    from utils.visualization import calculate_stability
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history.get('val_accuracy', [0])[-1]
    report = f"""
    ================ 模型性能报告 ================
    训练总轮次: {len(history.epoch)}
    训练效率:
    - 总训练时间: {sum(timer.times):.2f}秒
    - 平均每轮时间: {sum(timer.times)/len(timer.times):.2f}秒
    - 最快轮次: {min(timer.times):.2f}秒
    - 最慢轮次: {max(timer.times):.2f}秒
    收敛稳定性:
    - 最终训练损失: {history.history['loss'][-1]:.4f}
    - 训练损失方差(最后5轮): {calculate_stability(history.history['loss']):.6f}
    - 最终验证损失: {history.history.get('val_loss', ['N/A'])[-1]}
    - 验证损失方差(最后5轮): {calculate_stability(history.history.get('val_loss', [0])):.6f}
    准确率表现:
    - 最终训练准确率: {final_train_acc:.4f}
    - 最终验证准确率: {final_val_acc:.4f}
    ============================================
    """
    return report