from tensorflow.keras.callbacks import Callback
class CustomCallback(Callback):
    def __init__(self, target_accuracy):
        super(CustomCallback, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        if current_accuracy is not None and current_accuracy >= self.target_accuracy:
            print(f"\nTraining stopped as accuracy reached {self.target_accuracy}%.")
            self.model.stop_training = True