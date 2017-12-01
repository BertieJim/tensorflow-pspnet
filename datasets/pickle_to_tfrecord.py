#convert pickle satelite data into tfrecord format
import pandas as pd
import tensorflow as tf
import numpy as np

DATA_DIR = '../../satelite_data/u-net/train_pickle/'
_FILE_PATTERN = 'train_%s_batch_%d.pkl'
TF_RECORD = 'satelite_train.tfrecords'
def convert():
    writer = tf.python_io.TFRecordWriter(TF_RECORD)
    for batch in range(1,3):
        images = pd.read_pickle(DATA_DIR+_FILE_PATTERN%('X', batch))
        labels = pd.read_pickle(DATA_DIR+_FILE_PATTERN%('label', batch))
        print(images.shape, labels.shape)
        num_patterns, h, w = images.shape[:-1]
        labels = labels.reshape(num_patterns, h, w, 1)
        print(num_patterns)
        for i in range(num_patterns):
            print(type(images[i][0][0][0]), labels[i].shape)
            image = images[i].tostring()
            label = labels[i].tostring()
            print(type(image))
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))

            }))
            writer.write(example.SerializeToString())

    writer.close()
def create_fake():
    images = np.ones((20, 512,512,3))
    labels = np.ones((20,512,512))
    writer = tf.python_io.TFRecordWriter('fake.tfrecords')
    for i in range(20):
              image = images[i].tostring()
              label = labels[i].tostring()
              print(type(image))
              example = tf.train.Example(features=tf.train.Features(feature={
                  'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                  'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
              }))
              writer.write(example.SerializeToString())
    writer.close()
def read():

    from IPython import embed
    # # output file name string to a queue
    # filename_queue = tf.train.string_input_producer(['./satelite_train.tfrecord'], num_epochs=None)
    # # create a reader from file queue
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    # # get feature from serialized example
    #
    # features = tf.parse_single_example(serialized_example,
    #                                    features={
    #                                        'image': tf.FixedLenFeature([], tf.string),
    #                                        'label': tf.FixedLenFeature([], tf.string)                                       }
    #                                    )
    #
    # c_raw_out = features['image']
    # c_out = tf.decode_raw(c_raw_out, tf.float32)
    # print(c_out)
    # c_batch = tf.train.shuffle_batch([c_out], batch_size=1,
    #                                                    capacity=200, min_after_dequeue=100, num_threads=2)
    # sess = tf.Session()
    # init = tf.initialize_all_variables()
    # sess.run(init)
    #
    # tf.train.start_queue_runners(sess=sess)
    # c_val = sess.run([c_batch])
    # print(c_val)
    # 先定义feature，这里要和之前创建的时候保持一致
    feature = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string, default_value = 'str'),
        'label/encoded': tf.FixedLenFeature([], tf.string)
    }
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer(['satelite_train.tfrecords'])

    # 定义一个 reader ，读取下一个 record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 解析读入的一个record
    features = tf.parse_single_example(serialized_example, features=feature)

    # 将字符串解析成图像对应的像素组
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.decode_raw(features['label/encoded'], tf.uint8)
    format  = features['image/format']
    print(format)
    # 将标签转化成int32
    # label = tf.cast(features['label'], tf.float32)

    # 这里将图片还原成原来的维度
#    print(image, label)
    image = tf.reshape(image, [512, 512, 3])
    label = tf.reshape(label, [512, 512])
    # 你还可以进行其他一些预处理....

    # 这里是创建顺序随机 batches(函数不懂的自行百度)
    if True:
        images, labels, formats = tf.train.shuffle_batch([image, label, format], batch_size=2, capacity=20 * 512 * 512 * 3,
                                            min_after_dequeue=10)
    else:
        images, labels = image, label
    with tf.Session() as sess:

        # 初始化

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
 
        # 启动多线程处理输入数据
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img, lbl, format =sess.run([images, labels, formats])
        print(img.shape, lbl.shape)
        print(format)
        # 关闭线程
        coord.request_stop()
        coord.join(threads)
        sess.close()
def read_2():
  for serialized_example in tf.python_io.tf_record_iterator('fake.tfrecords'):
    # Get serialized example from file
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    label = example.features.feature["label"]
    features = example.features.feature["image"]
    print(type(label), type(features))
if  __name__ == '__main__':
     convert()
    # create_fake()
   # read()
