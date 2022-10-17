import threading
import gc
from queue import Queue
import pickle
import psutil
import tensorflow as tf
from tensorflow.python.framework.tensor_util import TensorShapeProtoToList
from Tensorflow.TFProcessImage import process_image
from config import train_config

def ParserRecord(np_array, labels):
  np_array = process_image.transform_images_setup(is_crop=False)(np_array)
  labels = tf.one_hot(labels, depth=train_config.num_classes)
  print(np_array)
  return np_array, labels

class DatasetQueue(object):
  def __init__(self, 
               datafile, 
               labelfile, 
               num_pbatch=65,
               num_epochs=10,
               num_pbatch_dataset = 10,
               queue_size = 2,
               batch_size = 32
               ):
    self.__datafile = datafile
    self.__labelfile = labelfile
    self.__num_pbatch = num_pbatch
    self.__num_epochs = num_epochs
    self.__queue_size = queue_size
    self.__num_pbatch_dataset = num_pbatch_dataset
    self.__batch_size = batch_size

    # property init
    self.__cur_iter = None
    self.__cur_dataset = None
    self.__dataQueue = Queue(maxsize=self.__queue_size)
    self.__isEnd = False
    self._lock = threading.Lock()

    # config
    self.__isMap = False
    self.__isRepeat = False
    self.__isShuffle = False
    self.__isPrefetch = False

    #==
    self.load_thread = threading.Thread(target=self.__load)
    # self.load_thread.start()

  def map(self, f):
    self.__isMap = True
    self.__map_f = f
    return self

  def repeat(self):
    self.__isRepeat = True
    return self

  def shuffle(self, buffer_size, seed=None):
    self.__isShuffle = True
    self.__shuffle_buffer_size = buffer_size
    self.__shuffle_seed = seed
    return self

  def prefetch(self, buffer_size):
    self.__isPrefetch = True
    self.__prefetch_buffer_size = buffer_size
    return self

  def batch(self, batch_size):
    self.__batch_size = batch_size
    return self

  def __load(self):
    for i_epoch in range(self.__num_epochs):
      print('DatasetQueue -- __load -- i_epoch : ', i_epoch)
      f_data = open(self.__datafile, 'rb')
      unpickler_data = pickle.Unpickler(f_data)

      f_labels = open(self.__labelfile, 'rb')
      unpickler_labels = pickle.Unpickler(f_labels)

      dataset = None
      for i_pbatch in range(self.__num_pbatch):
        print('    i_pbatch: ', i_pbatch, ' -- RAM: ', psutil.virtual_memory().percent)
        pbatch_data = unpickler_data.load()
        pbatch_labels = unpickler_labels.load()
        pbatch_dataset = tf.data.Dataset.from_tensor_slices((pbatch_data, pbatch_labels))
        
        #concat pbatch to dataset
        if dataset is None:
          dataset = pbatch_dataset
        dataset = dataset.concatenate(pbatch_dataset)

        # add dataset to queue
        if i_pbatch + 1 % self.__num_pbatch_dataset:
          # if self.__isShuffle:
          #   dataset.shuffle(self.__shuffle_buffer_size, seed = self.__shuffle_seed)
          # dataset.batch(self.__batch_size)
          # if self.__isMap:
          #   dataset.map(self.__map_f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
          # if self.__isPrefetch:
          #   dataset.prefetch(self.__prefetch_buffer_size)
          self.__dataQueue.put(dataset)
          dataset = None
        pass
      f_data.close()
      f_labels.close()
      pass

    with self._lock:
      self.__isEnd = True
    pass

  def getBatch(self):
    batch = None
    try:
        batch = next(self.__cur_iter)
        print('ddddddddddddddd', batch)
    except tf.errors.OutOfRangeError:
        del self.__cur_dataset
        gc.collect()
        if self.__isEnd:
          return -1
        self.__cur_dataset = self.__dataQueue.get()
        self.__cur_iter = iter(self.__cur_dataset)
    except:
      self.__cur_dataset = self.__dataQueue.get()
      self.__cur_dataset.map(ParserRecord).shuffle(10240, seed=43).repeat().batch(train_config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
      self.__cur_iter = iter(self.__cur_dataset)
      batch = next(self.__cur_iter)
      print('ddddddddddddddd', batch)

    return batch
