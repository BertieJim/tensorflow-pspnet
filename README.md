# tensorflow-pspnet

How to run pspnet with satelite data

1.datasets/pickle_to_tfrecord文件可以将pickle文件制作成tfrecord。指定DATA_DIR和_FILE_PATTERN。
pickle文件可以用之前上传到群文件的split_theone.py对图片进行切割，保存pkl文件。

2. 在datasets/satelite.py文件里修改 _FILE_PATTERN，即上一步生成的tfrecord的名称。

3. 修改train_pspnet.sh, 主要是DATASET_DIR，即tfrecord的位置，默认为／datasets／; 以及train_image_size，即在split的时候切割的图片的大小。

4.theoretically，有一个GPU服务器。基本就可以run了.
