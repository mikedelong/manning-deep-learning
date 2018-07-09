import logging
from os.path import isdir
from time import time

import numpy as np
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential


def vectorize_sequences(arg_sequences, arg_dimension=10000):
    size = (len(arg_sequences), arg_dimension)
    result = np.zeros(size)
    for index, sequence in enumerate(arg_sequences):
        result[index, sequence] = 1.0
    return result


if __name__ == '__main__':
    start_time = time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    verbose = 0
    output_folder = '../output/'

    output_folder_exists = isdir(output_folder)
    if not output_folder_exists:
        logger.warning('output folder %s does not exist. Quitting.' % output_folder)
        quit()

    num_words = 10000
    logger.debug('loading IMDB data, top %d most frequent words' % num_words)
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

    logger.debug('our first item looks like this: %s \nand its label is %s' % (train_data[0], train_labels[0]))
    logger.debug('the maximum word index is %d' % max([max(sequence) for sequence in train_data]))
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(index - 3, '?') for index in train_data[0]])
    logger.debug(decoded_review)

    x_train = vectorize_sequences(train_data)
    y_train = np.asarray(train_labels).astype('float32')
    x_test = vectorize_sequences(test_data)
    y_test = np.asarray(test_labels).astype('float32')
    logger.debug('sample data: %s' % x_train[0])

    model = Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    logger.debug('done')
    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
