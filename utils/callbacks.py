import time
import tensorflow as tf

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.times.append(epoch_time)
        print(f"Epoch {epoch+1}: {epoch_time:.2f}秒")