from keras.preprocessing.text import Tokenizer
import json
import os
import io
from data_generator import CharDataGenerator
from tensorflow.python.keras.layers import Input,Embedding,GRU,Dense,TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
import numpy as np

if __name__ == '__main__':
    #Define model parameters
    WHICH_GPU="1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = WHICH_GPU;

    BATCH_SIZE=1000
    TIMESTEPS=333
    HIDDEN_SIZE=128
    NUM_EPOCHS=10

    data=[]
    with open('data/paulgraham.txt') as f:
        for line in f:
            data.append(line.strip().lower())
    tokenizer=Tokenizer(num_words=None,char_level=True,filters=None,oov_token='UNK')
    tokenizer.fit_on_texts(data)

    train_data,val_data=train_test_split(data, test_size=0.05)
    training_generator=CharDataGenerator(train_data,tokenizer,timesteps=TIMESTEPS,hidden_size=HIDDEN_SIZE,batch_size=BATCH_SIZE,shuffle=True)
    validation_generator=CharDataGenerator(val_data,tokenizer,timesteps=TIMESTEPS,hidden_size=HIDDEN_SIZE,batch_size=BATCH_SIZE,shuffle=False)
    inputs = Input(shape=(None,len(tokenizer.word_index) + 1,), name='inputs')
    encoder_gru1 = GRU(HIDDEN_SIZE, return_sequences=True, return_state=True, name='encoder_gru1')
    encoder_out1, encoder_state1 = encoder_gru1(inputs)
    encoder_gru2 = GRU(HIDDEN_SIZE, return_sequences=True, return_state=True, name='encoder_gru2')
    encoder_out2, encoder_state2 = encoder_gru2(encoder_out1)

    # dense layer
    dense = Dense(len(tokenizer.word_index) + 1, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    output_pred = dense_time(encoder_out2)

    full_model = Model(inputs=inputs, outputs=output_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')
    full_model.summary(line_length=225)
    inf_state = Input(shape=(HIDDEN_SIZE,))
    pred_input=Input(shape=(None,len(tokenizer.word_index) + 1,), name='pred_inputs')
    pred_init_state1=Input(shape=(HIDDEN_SIZE,))
    pred_init_state2=Input(shape=(HIDDEN_SIZE,))
    pred_encoder_out1, pred_encoder_state1 = encoder_gru1(pred_input,initial_state=pred_init_state1)
    pred_encoder_out2, pred_encoder_state2 = encoder_gru2(pred_encoder_out1,initial_state=pred_init_state2)
    pred_output=dense_time(pred_encoder_out2)
    pred_model=Model(inputs=[pred_input,pred_init_state1,pred_init_state2],outputs=[pred_output,pred_encoder_state1,pred_encoder_state2])
    pred_model.summary(line_length=225)

    model_saver = ModelCheckpoint('h5.models/full_model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='auto', period=1)
    full_model.fit_generator(generator=training_generator, validation_data=validation_generator,
                             use_multiprocessing=True, workers=6, epochs=NUM_EPOCHS, callbacks=[model_saver])
    tokenizer_json = tokenizer.to_json()
    with io.open('h5.models/tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    full_model.save('h5.models/full_model.h5')

    #predict mode
    output_text=''
    text='t'
    initial_state1=np.zeros((1,HIDDEN_SIZE))
    initial_state2=np.zeros((1, HIDDEN_SIZE))
    bContinue=True
    total=0
    text_encoded = tokenizer.texts_to_sequences([text])
    preproc_text = pad_sequences(text_encoded, padding='post', maxlen=TIMESTEPS)
    while bContinue:
        preproc_categorical = to_categorical(preproc_text, num_classes=len(tokenizer.word_index) + 1)
        pred_val,current_state1,current_state2=pred_model.predict([preproc_categorical,initial_state1,initial_state2])
        index_value = np.argmax(pred_val, axis=-1)[0, 0]
        sTemp = tokenizer.index_word.get(index_value, 'UNK')
        print(sTemp)
        text+=sTemp
        total += 1
        if total >= TIMESTEPS or sTemp in '.?!':
            bContinue = False
        initial_state1=current_state1
        initial_state2=current_state2
        preproc_text[0, 0] = index_value
    print(text)