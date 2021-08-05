import os
from tensorflow.keras.preprocessing.text import Tokenizer

data_dir = 'data\\nlp_datasets\\gutenberg'
file_list = os.listdir(data_dir)

# get full path for all data files
i = 0
for file in file_list:
    full_path = os.path.join(data_dir, file)
    file_list[i] = full_path
    i = i + 1


files = []
for path in file_list:
    file = open(path, 'r').read()
    files.append(file)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(files)

print(len(tokenizer.word_index))
