from keras.preprocessing.text import Tokenizer
import io
import json
punctuation='.?!'

num_words=1
train_data=[]
with open('data/train.txt',encoding='utf-8') as f:
    for line in f:
        sText='startsentence '+line.lower().strip().replace('"','')
        if sText[-1] in punctuation:
            sText+=' endsentence'
        else:
            sText+='. endsentence'
        train_data.append(sText.lower().strip())

tokenizer=Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(train_data)
tokenizer_json=tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))