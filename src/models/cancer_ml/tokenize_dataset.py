from tensorflow.keras.preprocessing.text import Tokenizer

def tokenize_dataset(pd_dataset):

    # replace missing spots with a string then after dataset is encoded, replace with mean of column
    df = pd_dataset.fillna('empty')

    if df.shape[0] >= df.shape[1]:
        long_axis = df.shape[0]
        short_axis = df.shape[1]
    else:
        long_axis = df.shape[1]
        short_axis = df.shape[0]

    word_list = []
    for i in range(long_axis):
        for n in range(short_axis):
            
            if long_axis == df.shape[0]:
                data = pd_dataset.iloc[i, n]
            else:
                data = pd_dataset.iloc[n, i]

            if str(type(data)) == "<class 'str'>":

                # list of chars to be removed from data
                char_blocked = [' ', '.', '/', '-', '_', '>', '+', ',', ')', '(', '*',
                                '=', '?', ':', '[', ']', '#', '!', '\n', '\\', '}',
                                '{', ';', '%', '"']
                
                for char in char_blocked:
                    if char in data:
                        data = data.replace(char, '')

                if long_axis == df.shape[0]:
                    pd_dataset.iloc[i, n] = data
                else:
                    pd_dataset.iloc[n, i] = data

                word_list.append(data)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(word_list)
        code_dict = tokenizer.word_index

        for i in range(long_axis):
            for n in range(short_axis):
                if long_axis == df.shape[0]:
                    data = pd_dataset.iloc[i, n]
                else: 
                    data = pd_dataset.iloc[n, i]

                if str(type(data)) == "<class 'str'>":

                    data = data.lower()
                    data = int(code_dict[data])

                if long_axis == df.shape[0]: 
                    pd_dataset.iloc[i, n] = data
                else: 
                    pd_dataset.iloc[n, i] = data

        # convert all cols to numeric vals
        pd_dataset = pd_dataset.astype('int64')

        # replace spots previously denoted as 'empty' with mean of column
        for column in list(pd_dataset.columns):
            col = pd_dataset[column]

            i = 0
            for val in col:
                if val == code_dict['empty']:
                    col[i] = pd_dataset[column].mean()

                i = i + 1
            
            pd_dataset[column] = col

    return pd_dataset
    