from __future__ import unicode_literals, print_function, division

import torch
import unicodedata
import string
from sklearn.model_selection import train_test_split

all_letters = string.ascii_letters + " .,;'<"
n_letters = len(all_letters)+1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def letterToIndex(letter):
    return all_letters.find(letter)

def indexToLetter(index):
    return all_letters[index]

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def tensor_to_line(tensor):
    line=''
    for one_tensor in tensor:
        index=torch.nonzero(one_tensor)[0][1]
        line=line+(all_letters[index])
    return line


def tts(df, perc):
    train_clean_df, test_clean_df = train_test_split(df, test_size=perc)

    return train_clean_df, test_clean_df

def pad_word(word,max_length=20):
    if len(word)<max_length:
        padded_word=word+('<'*(max_length-len(word)))
    else:
        padded_word = word[0:max_length]
    return padded_word
