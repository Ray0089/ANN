import numpy as np
import struct
import matplotlib.pyplot as plt
import os

###
#you can download the mnist dataset from http://yann.lecun.com/exdb/mnist/
###
def read_train_data(path=''):
    with open(os.path.join(path, 'train-images.idx3-ubyte'), 'rb') as f1:
        buf1 = f1.read()
    with open(os.path.join(path, 'train-labels.idx1-ubyte'), 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def read_test_data(path=''):
    with open(os.path.join(path, 't10k-images.idx3-ubyte'), 'rb') as f1:
        buf1 = f1.read()
    with open(os.path.join(path, 't10k-labels.idx1-ubyte'), 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def get_image(buf1):
    image_index = 0
    image_index += struct.calcsize('>IIII')
    img_num = int((len(buf1) - 16) / 784)
    im = []
    for i in range(img_num):
        temp = list(struct.unpack_from('>784B', buf1, image_index)) # '>784B'的意思就是用大端法读取784个unsigned byte
        im.append(temp)
        image_index += struct.calcsize('>784B')  # 每次增加784B
    im = np.array(im, dtype=np.float32)
    return im


def get_label(buf2): # 得到标签数据
    label_index = 0
    label_index += struct.calcsize('>II')
    idx_num = int(len(buf2) - 8)
    labels = []
    for i in range(idx_num):
        temp = list(struct.unpack_from('>1B', buf2, label_index))
        labels.append(temp)
        label_index += 1
    labels = np.array(labels, dtype=np.int)
    return labels


def load_train_data(path=''):
    img_buf, label_buf = read_train_data(path)
    imgs = get_image(img_buf)
    labels = get_label(label_buf)

    return imgs, labels


def load_test_data(path=''):
    img_buf, label_buf = read_test_data(path)
    imgs = get_image(img_buf)
    labels = get_label(label_buf)

    return imgs, labels


if __name__ == "__main__":

    imgs, labels = load_test_data()

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        title = u"标签对应为：" + str(labels[i])
        plt.title(title, fontproperties='SimHei')
        img = np.array(imgs[i]).reshape((28, 28))
        plt.imshow(img, cmap='gray')
    plt.show()