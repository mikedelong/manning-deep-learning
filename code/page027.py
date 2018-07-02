import logging
import time

from keras.datasets import mnist

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

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    logger.debug('the training data has shape %d x %d x %d' % train_images.shape)
    logger.debug('the training labels look like this: %s' % train_labels)
    logger.debug('the test data has shape %d x %d x %d' % test_images.shape)
    logger.debug('the test labels look like this: %s' % test_labels)


    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
