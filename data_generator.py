from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn.utils import shuffle
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

class DataGenerator(Sequence):
    def __init__(self,text,tokenizer,timesteps,batch_size=32,shuffle=True):
        self.text=text
        self.tokenizer=tokenizer
        self.timesteps=timesteps
        self.timesteps=timesteps
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.text)/float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle==True:
            self.text=shuffle(self.text)

    def __getitem__(self,idx):
        text=self.text[idx * self.batch_size:(idx + 1) * self.batch_size]
        text_encoded = self.tokenizer.texts_to_sequences(text)
        preproc_text = pad_sequences(text_encoded, padding='post', maxlen=self.timesteps)
        target_categorical=to_categorical(preproc_text,num_classes=len(self.tokenizer.word_index))
        return preproc_text[:,:-1],target_categorical[:,1:,:]