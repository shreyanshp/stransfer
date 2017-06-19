import os
import numpy as np
from os.path import exists
from sys import stdout
import tensorflow as tf
import transform
import scipy
import scipy.misc
import base64
from PIL import Image
from io import BytesIO

def strans(in_img, mynetwork):
    network = mynetwork
    if not os.path.isdir(network):
        raise Exception("Network %s does not exist." % network)

    content_image = scipy.misc.imread(in_img, mode="F")
    if (len(content_image.shape) != 3) or (content_image.shape[2] != 3):
        content_image = np.dstack((content_image, content_image, content_image))

    content_image = content_image.astype("float32")
    content_image = np.ndarray.reshape(content_image, (1,) + content_image.shape)

    mprediction = ffwd(content_image, network)

    savedat = np.clip(mprediction, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(savedat)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    return new_image_string


def ffwd(content, network_path):
    with tf.Graph().as_default(), tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=content.shape,
                                         name='img_placeholder')

        network = transform.net(img_placeholder)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(network_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")

        prediction = sess.run(network, feed_dict={img_placeholder:content})
        return prediction[0]