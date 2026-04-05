import numpy as np
import json
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.data_loader import reshape_samples
from utils.data_preprocessor import preparing
from utils.callbacks import TimingCallback
from utils.performance_report import generate_performance_report
from utils.visualization import plot_model_comparison

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
    checkpointer = ModelCheckpoint(filepath=save_path+'/best_model.keras', verbose=1, save_best_only=True)
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

def save_model(save_path,model,translator):
    model_file=save_path+'/classifier_model.keras'
    model.save(model_file)
    json_str = json.dumps(translator)
    with open(save_path+'/translator.json', 'w') as f:
        f.write(json_str)
    print(f'Save model sucessfully!\nSaving path:{save_path}')
    return True