# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import numpy as np
import json
import MySQLdb
from sklearn import preprocessing
from bs4 import BeautifulSoup
from langdetect import detect
import spacy
import en_core_web_sm
import sys
import pickle
import os


def load_yelp(alphabet):
    examples = []
    labels = []
    db = MySQLdb.connect("10.249.71.213", "root", "root", "ai")
    cursor = db.cursor()
    previous_y = []
    sql = "SELECT DISTINCT(sr_number),t1_final,t2_final ,subject,body FROM nice_text_source_data WHERE t2_final in ('Defect Appeal', \
              'High Risk','Site Features - CCR','Selling Performance','VeRO - CCR','Bidding/Buying Items','Report a Member/Listing','Account Restriction', \
              'Cancel Transaction','Logistics - CCR','Selling Limits - CCR','Listing Queries - CCR','Paying for Items','Seller Risk Management', \
              'eBay Account Information - CCR','Shipping - CCR','Account Suspension','Buyer Protection Case Qs','Buyer Protect High ASP Claim', \
              'Buyer Protection Appeal INR','eBay Fees - CCR','Completing a Sale - CCR')"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()

    except:
        sys.stdout.write("Error: unable to fecth data"+ '\n')

    db.close()
    i=0
    for row in results:
        text = (row[3] + ' ' + row[4]).decode('utf8', 'ignore')

        try:
            if text!='' and detect(text)=='en':
                text_end_extracted = extract_end(list(row[3] + ' ' + row[4]))
                padded = pad_sentence(text_end_extracted)
                text_int8_repr = string_to_int8_conversion(padded, alphabet)
                previous_y.append(row[1] + '|' + row[2])
                examples.append(text_int8_repr)
                #x_text.append(clean_str(text))
                i=i+1
                sys.stdout.write("Value is %s" % i)
                sys.stdout.write('\n')
                #previous_y.append(row[1] + '|' + row[2])
        except:
            sys.stdout.write(row[0])

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(list(previous_y))

    f = open('y_target.pickle', 'wb')
    pickle.dump(lb, f)
    f.close()

    sys.stdout.write(lb.inverse_transform(labels))

    return examples, labels


def extract_end(char_seq):
    if len(char_seq) > 20000:
        char_seq = char_seq[-20000:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    char_seq_length = 20000
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x


def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    x_batch = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]
    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def load_data():
    # TODO Add the new line character later for the yelp'cause it's a multi-line review
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    examples, labels = load_yelp(alphabet)
    x = np.array(examples, dtype=np.int8)
    y = np.array(labels, dtype=np.int8)
    print("x_char_seq_ind=" + str(x.shape))
    print("y shape=" + str(y.shape))
    return [x, y]


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
            batch = list(zip(x_batch, y_batch))
            yield batch
