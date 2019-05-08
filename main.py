from keras.preprocessing.text import Tokenizer
from data_generator import DataGenerator
from tensorflow.python.keras.layers import Input,Embedding,GRU,Dense,TimeDistributed
from tensorflow.python.keras.models import Model
import json
import os
import io

#Define model parameters
WHICH_GPU="1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = WHICH_GPU;


BATCH_SIZE=32
EMBEDDING_DIM=100
TIMESTEPS=22
HIDDEN_SIZE=128
NUM_EPOCHS=10
punctuation='.?!'

train_data=[]
with open('data/train.txt',encoding='utf-8') as f:
    for line in f:
        sText='startsentence '+line.lower().strip().replace('"','')
        if sText[-1] in punctuation:
            sText+=' endsentence'
        else:
            sText+='. endsentence'
        train_data.append(sText.lower().strip())
test_data=[]
with open('data/test.txt',encoding='utf-8') as f:
    for line in f:
        sText='startsentence '+line.lower().strip().replace('"','')
        if sText[-1] in punctuation:
            sText+=' endsentence'
        else:
            sText+='. endsentence'
        test_data.append(sText.lower().strip())

tokenizer=Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(train_data)

training_generator=DataGenerator(train_data,tokenizer,timesteps=TIMESTEPS,batch_size=BATCH_SIZE,shuffle=True)
validation_generator=DataGenerator(test_data,tokenizer,timesteps=TIMESTEPS,batch_size=BATCH_SIZE,shuffle=False)
inputs=Input(shape=(TIMESTEPS-1,),name='inputs')
#embedding_layer=Embedding(input_dim=len(tokenizer.word_index),output_dim=EMBEDDING_DIM,input_length=TIMESTEPS,weights=[emb_matrix],trainable=False,name='embedding_layer')
embedding_layer = Embedding(input_dim=len(tokenizer.word_index), output_dim=EMBEDDING_DIM)
input_embedded=embedding_layer(inputs)

encoder_gru1 = GRU(HIDDEN_SIZE, return_sequences=True, return_state=True, name='encoder_gru1')
encoder_out1, encoder_state1 = encoder_gru1(input_embedded)
encoder_gru2 = GRU(HIDDEN_SIZE, return_sequences=True, return_state=True, name='encoder_gru2')
encoder_out2, encoder_state2 = encoder_gru2(encoder_out1)


# dense layer
dense = Dense(len(tokenizer.word_index)+1, activation='softmax', name='softmax_layer')
dense_time = TimeDistributed(dense, name='time_distributed_layer')
output_pred = dense_time(encoder_out2)

full_model = Model(inputs=inputs, outputs=output_pred)
full_model.compile(optimizer='adam', loss='categorical_crossentropy')
full_model.summary(line_length=225)
full_model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=6,epochs=NUM_EPOCHS)
tokenizer_json=tokenizer.to_json()
with io.open('h5.models/tokenizer.json','w',encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json,ensure_ascii=False))
full_model.save('h5.models/full_model.h5')