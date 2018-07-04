import logging
import time
from os.path import isdir

import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    output_folder = '../output/'

    output_folder_exists = isdir(output_folder)
    if not output_folder_exists:
        logger.warning('output folder %s does not exist. Quitting.' % output_folder)
        quit()

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    logger.debug('the training data has shape %d x %d x %d' % train_images.shape)
    logger.debug('the training labels look like this: %s' % train_labels)
    logger.debug('the test data has shape %d x %d x %d' % test_images.shape)
    logger.debug('the test labels look like this: %s' % test_labels)

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    digit = train_images[4].reshape((28, 28))
    logger.debug(digit.shape)
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.savefig(output_folder + 'page027-4th-digit.png')

    network.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    test_loss, test_accuracy = network.evaluate(test_images, test_labels, verbose=0)
    logger.debug('test accuracy: %.4f' % test_accuracy)

    digit = train_images[4].reshape((28, 28))
    logger.debug(digit.shape)
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.savefig(output_folder + 'page027-4th-digit.png')

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
