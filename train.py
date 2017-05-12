import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import random
import numpy
import cv2

tf.app.flags.DEFINE_string("mode", "test", "train or test")
tf.app.flags.DEFINE_string("checkpoint", "./checkpoint/", "dir of checkpoint")
tf.app.flags.DEFINE_string("train_dir", "c:/Datasets/HWDB/trn", "dir of training data")
tf.app.flags.DEFINE_string("test_dir", "c:/Datasets/HWDB/tst", "dir of test data")
tf.app.flags.DEFINE_string("test_image", "", "test image for pictest")
tf.app.flags.DEFINE_string("logger_dir", "./logger", "dir of logger")
tf.app.flags.DEFINE_integer("batch_size", 128, "size of batch")
tf.app.flags.DEFINE_integer("img_size", 64, "size of resized images")
tf.app.flags.DEFINE_string("char_dict", "char_dict", "path to character dict")
tf.app.flags.DEFINE_bool("restore", False, "restore from previous checkpoint")
tf.app.flags.DEFINE_integer("max_step", 100001, "maximum steps")
tf.app.flags.DEFINE_string("test_pic", "./test.png", "path to test picture")
FLAGS = tf.app.flags.FLAGS


# Read data from directory
def get_image_path_and_labels(dir):
    img_path = []
    for root, dir, files in os.walk(dir):
        img_path += [os.path.join(root, f) for f in files]
    # Shuffle the data to avoid overfit
    random.shuffle(img_path)
    # Because I created folders corresponding to each character, so the folder name is actual label
    labels = [int(name.split(os.sep)[len(name.split(os.sep)) - 2]) for name in img_path]
    return img_path, labels


def batch(dir, batch_size, prepocess=False):
    img_path, labels = get_image_path_and_labels(dir)
    # Transfer data into tensor so that it could be feed into network.
    img_tensor = tf.convert_to_tensor(img_path, dtype=tf.string)
    lb_tensor = tf.convert_to_tensor(labels, dtype=tf.int64)
    # Loading the entire dataset comes with extremely high cost on RAM, so slice it and read them batch by batch
    input_pipe = tf.train.slice_input_producer([img_tensor, lb_tensor])
    # Read actual data and turn them to grey scale
    img = tf.read_file(input_pipe[0])
    imgs = tf.image.convert_image_dtype(tf.image.decode_png(img, channels=1), tf.float32)
    # Randomly modify the images in order to avoid overfit.
    if prepocess:
        imgs = tf.image.random_contrast(imgs, 0.9, 1.1)
    # Resize original images to unified size.
    imgs = tf.image.resize_images(imgs, tf.constant([FLAGS.img_size, FLAGS.img_size], dtype=tf.int32))
    # Read label
    lbs = input_pipe[1]
    # Get batch
    img_batch, lb_batch = tf.train.shuffle_batch([imgs, lbs], batch_size=batch_size, capacity=50000,
                                                 min_after_dequeue=10000)
    return img_batch, lb_batch


def cnn():
    # (1-keep_prob) equals to dropout rate on fully-connected layers
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    # Set up places for data and label, so that we can feed data into network later
    img = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="img_batch")
    labels = tf.placeholder(tf.int64, shape=[None], name="label_batch")
    # Structure references to : http://yuhao.im/files/Zhang_CNNChar.pdf,
    # however I adjust a little bit due to limited computational resource.
    # Four convolutional layers with kernel size of [3,3], and ReLu as activation function
    conv1 = slim.conv2d(img, 64, [3, 3], 1, padding="SAME", scope="conv1")
    pool1 = slim.max_pool2d(conv1, [2, 2], [2, 2], padding="SAME")
    conv2 = slim.conv2d(pool1, 128, [3, 3], padding="SAME", scope="conv2")
    pool2 = slim.max_pool2d(conv2, [2, 2], [2, 2], padding="SAME")
    conv3 = slim.conv2d(pool2, 256, [3, 3], padding="SAME", scope="conv3")
    pool3 = slim.max_pool2d(conv3, [2, 2], [2, 2], padding="SAME")
    conv4 = slim.conv2d(pool3, 512, [3, 3], [2, 2], scope="conv4", padding="SAME")
    pool4 = slim.max_pool2d(conv4, [2, 2], [2, 2], padding="SAME")
    # Flat the feature map so that we can connect it to fully-connected layers
    flat = slim.flatten(pool4)
    # Two fully-connected layers with dropout rate as mentioned at the start
    # First layer used tanh() as activation function
    fcnet1 = slim.fully_connected(slim.dropout(flat, keep_prob=keep_prob), 1024, activation_fn=tf.nn.tanh,
                                  scope="fcnet1")
    fcnet2 = slim.fully_connected(slim.dropout(fcnet1, keep_prob=keep_prob), 3755, activation_fn=None, scope="fcnet2")
    # loss function is defined as cross entropy on result of softmax function on last layer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fcnet2, labels=labels))
    # compare result to actual label to get accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fcnet2, 1), labels), tf.float32))
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    # learning rate with exponential decay
    lrate = tf.train.exponential_decay(2e-4, step, decay_rate=0.97, decay_steps=2000, staircase=True)
    # Adam optimizer to decrease loss value
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss, global_step=step)
    prob_dist = tf.nn.softmax(fcnet2)
    val_top3, index_top3 = tf.nn.top_k(prob_dist, 3)
    # Write log into TensorBoard
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    summary = tf.summary.merge_all()
    return {"img": img,
            "label": labels,
            "global_step": step,
            "optimizer": optimizer,
            "loss": loss,
            "accuracy": accuracy,
            "summary": summary,
            'keep_prob': keep_prob,
            "val_top3": val_top3,
            "index_top3": index_top3
            }


def train():
    with tf.Session() as sess:
        print("Start reading data")
        # Get batch tensor of data
        trn_imgs, trn_labels = batch(FLAGS.train_dir, FLAGS.batch_size, prepocess=True)
        tst_imgs, tst_labels = batch(FLAGS.test_dir, FLAGS.batch_size)
        graph = cnn()
        # Preparation before training
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        saver = tf.train.Saver()
        if not os.path.isdir(FLAGS.logger_dir):
            os.mkdir(FLAGS.logger_dir)
        trn_summary = tf.summary.FileWriter(os.path.join(FLAGS.logger_dir, 'trn'), sess.graph)
        tst_summary = tf.summary.FileWriter(os.path.join(FLAGS.logger_dir, 'tst'))
        step = 0
        # If received restore flag, train from last checkpoint
        if FLAGS.restore:
            # Get last checkpoint in checkpoint directory
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
            if checkpoint:
                # Restore data from checkpoint
                saver.restore(sess, checkpoint)
                step += int(checkpoint.split('-')[-1])
                print("Train from checkpoint")
        print("Start training")
        while not coord.should_stop():
            # Get actual data
            trn_img_batch, trn_label_batch = sess.run([trn_imgs, trn_labels])
            # Prepare parameters for network
            graph_dict = {graph['img']: trn_img_batch, graph['label']: trn_label_batch, graph['keep_prob']: 0.8}
            # Feed and parameter into network
            opt, loss, summary, step = sess.run(
                [graph['optimizer'], graph['loss'], graph['summary'], graph['global_step']], feed_dict=graph_dict)
            trn_summary.add_summary(summary, step)
            print("# " + str(step) + " with loss " + str(loss))
            if step > FLAGS.max_step:
                break
            # Evaluate current network based on test dataset
            if (step % 500 == 0) and (step >= 500):
                tst_img_batch, tst_label_batch = sess.run([tst_imgs, tst_labels])
                graph_dict = {graph['img']: tst_img_batch, graph['label']: tst_label_batch, graph['keep_prob']: 1.0}
                accuracy, test_summary = sess.run([graph['accuracy'], graph['summary']], feed_dict=graph_dict)
                tst_summary.add_summary(test_summary, step)
                print("Accuracy: %.8f" % accuracy)
                # Save checkpoint
                if step % 10000 == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoint, 'hccr'), global_step=graph['global_step'])
        coord.join(threads)
        saver.save(sess, os.path.join(FLAGS.checkpoint, 'hccr'), global_step=graph['global_step'])
        sess.close()
    return


def test(path):
    # Read test picture and resize it, turn it to grey scale.
    tst_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    tst_image = cv2.resize(tst_image,(64, 64))
    tst_image = numpy.asarray(tst_image) / 255.0
    tst_image = tst_image.reshape([-1, 64, 64, 1])
    # feed the test picture into network and estimate probability distribution
    with tf.Session() as sess:
        graph = cnn()
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(FLAGS.checkpoint))
        graph_dict = {graph['img']: tst_image, graph['keep_prob']: 1.0}
        val, index = sess.run([graph['val_top3'], graph['index_top3']], feed_dict=graph_dict)
        for i in range(3):
            print("Probability: %.5f"%val[0][i]+ " with label:"+str(index[0][i]))
        path=FLAGS.train_dir+"/" + '%0.5d' % index[0][0]
        # select one of the picture from the label with top 1 probability
        for root,dir,files in os.walk(path):
            for f in files:
                img=cv2.imread(path+"/"+f)
                enlarged=cv2.resize(img,(img.shape[1]*5,img.shape[0]*5))
                cv2.imshow("Top1",enlarged)
                cv2.waitKey()
                break
            break
    return val, index


def main(_):
    if FLAGS.mode == "train":
        train()
    if FLAGS.mode == "test":
        test(FLAGS.test_pic)


if __name__ == '__main__':
    tf.app.run()
