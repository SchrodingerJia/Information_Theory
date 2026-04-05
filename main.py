import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, SimpleRNN, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, PReLU, LeakyReLU, AveragePooling2D, Flatten, Dense, Dropout, SpatialDropout2D, SpatialDropout1D  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.constraints import max_norm # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
# 创建计时回调
class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.times.append(epoch_time)
        print(f"Epoch {epoch+1}: {epoch_time:.2f}秒")
#EMG及labels数据读取
def load_samples(filename):
    EMG=np.load(filename)['EMG']
    labels=np.load(filename)['labels']
    print('Load datas sucessfully!')
    samples=(EMG,labels)
    return samples
#获得众数
def mode(array):
    count = np.count_nonzero(array == 'IDLE')
    if count>=int(array.shape[0]):
        return 'IDLE'
    else:
        vals, counts = np.unique(array[array!='IDLE'], return_counts=True)
        index = np.argmax(counts)
        return vals[index]
#数据重构
def reshape_samples(samples,length=50):
    EMGs,labels=samples
    reshape_EMGs=np.array([EMGs[:,0:length]])
    reshape_labels=np.array([mode(labels[0:length])])
    for i in range(labels.shape[0]-int(length/10)):
        label=np.array([mode(labels[i+1:i+1+int(length/10)])])
        EMG=np.array([EMGs[:,i*10+10:i*10+10+length]])
        reshape_labels=np.concatenate((reshape_labels,label),axis=0)
        reshape_EMGs=np.concatenate((reshape_EMGs,EMG),axis=0)
    reshape_samples=(reshape_EMGs,reshape_labels)
    return reshape_samples
#数据预处理
def preparing(samples,test_size=0.2):
    X,y=samples
    label_set=set(y)
    label_dic={}
    for i in range(len(label_set)):
        label_dic[list(label_set)[i]]=i
    for j in range(len(y)):
        y[j]=label_dic[y[j]]
    y_categorical = to_categorical(y)  # one-hot 编码标签
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=28)
    return X_train, X_test, y_train, y_test, label_dic
def CNN_model(nb_classes, Chans=8, Samples=50, dropoutRate=0.4, kernLength=24, F1=12, D=1, F2=24, norm_rate=0.75, dropoutType = Dropout):
    input1 = Input(shape=(Chans, Samples, 1))
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=D)(block1)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    block1 = AveragePooling2D((1, 2))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    block2 = Conv2D(F2, (1, 8), padding='same', use_bias=False)(block1)
    block2 = BatchNormalization()(block2)
    block2 = LeakyReLU()(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate), activation='softmax')(flatten)
    return Model(inputs=input1, outputs=dense)
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
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
# 计算收敛稳定性指标
def calculate_stability(loss_values, window=5):
    """计算最后N个epoch的损失方差"""
    last_n_losses = loss_values[-window:]
    return np.var(last_n_losses)
def generate_performance_report(history, timer):
    """生成完整性能报告"""
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
#神经网络学习
def Learning(samples,model_type,save_path,sample_length):
    # 数据预处理
    samples=reshape_samples(samples,length=sample_length)
    X_train, X_test, y_train, y_test, label_dic=preparing(samples)
    classes_num=len(label_dic)
    trans_dic={value: key for key, value in label_dic.items()}
    # 构建模型
    model_cro = model_type(nb_classes=classes_num)
    model_mse = model_type(nb_classes=classes_num)
    # 编译模型
    model_cro.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    model_mse.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])
    # 定义回调函数
    checkpointer = ModelCheckpoint(filepath=save_path+'\\best_model.keras', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    timer_cro = TimingCallback()
    timer_mse = TimingCallback()
    # 训练模型
    history_cro = model_cro.fit(X_train, y_train, batch_size=16, epochs=200, verbose=1,
                        validation_split=0.2,  # 假设20%的数据用于验证
                        callbacks=[checkpointer, early_stopping,timer_cro])
    history_mse = model_mse.fit(X_train, y_train, batch_size=16, epochs=200, verbose=1,
                        validation_split=0.2,  # 假设20%的数据用于验证
                        callbacks=[checkpointer, early_stopping,timer_mse])
    # 评估模型
    y_pred = model_cro.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    #使用classification_report来获取每个类别的准确率和其他指标
    report = classification_report(y_true, y_pred_classes, target_names=[f'Class {trans_dic[i]}' for i in range(classes_num)])
    print(report)
    y_pred = model_mse.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    #使用classification_report来获取每个类别的准确率和其他指标
    report = classification_report(y_true, y_pred_classes, target_names=[f'Class {trans_dic[i]}' for i in range(classes_num)])
    print(report)
    print(generate_performance_report(history_cro, timer_cro))
    print(generate_performance_report(history_mse, timer_mse))
    plot_model_comparison(history_cro, history_mse)
    return model_cro,trans_dic
#模型保存
def save_model(save_path,model,translator):
    model_file=save_path+'\\classifier_model.keras'
    model.save(model_file)
    json_str = json.dumps(translator)
    with open(save_path+'\\translator.json', 'w') as f:
        f.write(json_str)
    print(f'Save model sucessfully!\nSaving path:{save_path}')
    return True
if __name__ == '__main__':
    save_path='./results'
    samples_file='./data/Samples.npz'
    sample_length=50
    model_type=CNN_model
    samples=load_samples(samples_file)
    model,translator=Learning(samples,model_type,save_path,sample_length)