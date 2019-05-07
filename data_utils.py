from keras.preprocessing.text import Tokenizer
import io
import json


def generate_tokenizer(filename):
    punctuation='.?!'

    num_words=1
    train_data=[]
    with open(filename,encoding='utf-8') as f:
        for line in f:
            sText='startsentence '+line.lower().strip().replace('"','')
            if sText[-1] in punctuation:
                sText+=' endsentence'
            else:
                sText+='. endsentence'
            train_data.append(sText)

    tokenizer=Tokenizer(oov_token='UNK')
    tokenizer.fit_on_texts(train_data)
    tokenizer_json=tokenizer.to_json()
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))