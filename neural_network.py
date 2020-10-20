import numpy as np
import os
import dataloader as dl
import random
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='aaa')
args = parser.parse_args()
data_path = args.data_path
###
# 激活函数
###

class Sigmoid(object):
    def __init__(self):
        self.gradient = []

    def forward(self, x):
        self.gradient = x * (1.0 - x)
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self):
        return self.gradient


class ReLU(object):

    def __init__(self):
        self.gradient = []

    def forward(self, input_data):
        self.gradient = np.where(input_data >= 0, 1, 0.001)
        self.gradient = self.gradient[:, None]
        input_data[input_data < 0] = 0.01*input_data[input_data < 0]
        return input_data

    def backward(self):
        return self.gradient

def softmax(input_data):
    # 减去最大值防止softmax上下溢出
    input_max = np.max(input_data)
    input_data -= input_max
    input_data = np.exp(input_data)
    exp_sum = np.sum(input_data)
    input_data /= exp_sum
    return input_data



###
# 全连接层
###

class FullyConnectedLayer(object):

    def __init__(self, input_size, output_size, learning_rate=0.03):
        self._w = np.random.randn(input_size*output_size)/np.sqrt(input_size*output_size)
        self._w = np.reshape(self._w, (input_size, output_size))
        b = np.zeros((1, output_size), dtype=np.float32)
        self._w = np.concatenate((self._w, b), axis=0)
        self._w = self._w.astype(np.float32)
        #self._w = np.ones((input_size + 1, output_size), dtype=np.float32)
        self.lr = learning_rate
        self.gradient = np.zeros((input_size + 1, output_size), dtype=np.float32)
        self.w_gradient = []
        self.input = []

    def forward(self, input_data):

        # put b into w matrix
        input_data = np.append(input_data, [1.0], axis=0)
        input_data = input_data.astype(np.float32)
        # calculate linear product
        output_data = np.dot(input_data.T, self._w)
        # save input data for gradient calculation
        self.input = input_data

        #update gradient
        self.gradient = self._w
        self.w_gradient = input_data

        return output_data

    def backward(self):
        return self._w#[:-1, :]

    def update(self, delta_grad):
        self.input = self.input[:, None]
        self._w -= self.lr * np.matmul(self.input, delta_grad)

    def get_w(self):
        return self._w

    def set_w(self, w):
        self._w = w


###
# CrossEntropyWithLogit 损失函数
###

class CrossEntropyWithLogit(object):

    def __init__(self):
        self.gradient = []

    def calculate_loss(self, input_data, y_gt):

        input_data = softmax(input_data)
        # 交叉熵公式 -sum(yi*logP(i))
        loss = -np.sum(y_gt * np.log(input_data + 1e-5))
        # calculate gradient
        self.gradient = input_data - y_gt

        return loss

    def predict(self, input_data):

        input_data = softmax(input_data)

        return np.argmax(input_data)

    def backward(self):
        return self.gradient.T


class MNISTNet(object):

    def __init__(self):
        self.linear_layer1 = FullyConnectedLayer(28*28, 128)
        self.linear_layer2 = FullyConnectedLayer(128, 10)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.loss = CrossEntropyWithLogit()

    def train(self, x, y):

        # forward
        x = self.linear_layer1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear_layer2.forward(x)
        x = self.relu2.forward(x)
        loss = self.loss.calculate_loss(x, y)
        #print("loss:{}".format(loss))
        # backward

        loss_grad = self.loss.backward()
        relu2_grad = self.relu2.backward()
        layer2_grad = self.linear_layer2.backward()
        grads = np.multiply(loss_grad, relu2_grad)
        self.linear_layer2.update(grads.T)
        grads = layer2_grad.dot(grads)
        relu1_grad = self.relu1.backward()
        grads = np.multiply(relu1_grad, grads)
        self.linear_layer1.update(grads.T)

        return loss

    def predict(self, x):
        # forward
        x = self.linear_layer1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear_layer2.forward(x)
        x = self.relu2.forward(x)
        number_index = self.loss.predict(x)

        return number_index

    def save(self, path='.', w1_name='w1', w2_name='w2'):

        w1 = self.linear_layer1.get_w()
        w2 = self.linear_layer2.get_w()
        np.save(os.path.join(path, w1_name), w1)
        np.save(os.path.join(path, w2_name), w2)

    def evaluate(self, x, y):
        if y == self.predict(x):
            return True
        else:
            return False

    def load_param(self, path=""):

        w1 = np.load(os.path.join(path,'w1.npy'))
        w2 = np.load(os.path.join(path,'w2.npy'))
        self.linear_layer1.set_w(w1)
        self.linear_layer2.set_w(w2)


def one_hot_encoding(y):
    one_hot_y = np.eye(10)[y]

    return one_hot_y


def train_net(data_path=''):

    m_net = MNISTNet()

    x_train, y_train = dl.load_train_data(data_path)
    x_train = x_train / 255 - 0.5
    y_train = one_hot_encoding(y_train)

    epoch = 20
    for i in range(epoch):
        average_loss = 0
        for j in range(x_train.shape[0]):
            average_loss += m_net.train(x_train[j], y_train[j])
            if j%2000 == 0:
                print('train set loss(epo:{}): {}'.format(i, average_loss / (j+1)))
        print('train set average loss: {}'.format(average_loss/x_train.shape[0]))
        m_net.save()


def eval_net(path=""):

    x_test, y_test = dl.load_test_data(path)
    x_test = x_test / 255.0 - 0.5
    precision = 0
    m_net = MNISTNet()
    m_net.load_param()
    for i in range(x_test.shape[0]):
        if m_net.evaluate(x_test[i], y_test[i]):
            precision += 1
    precision /= len(x_test)
    print('precision of test data set is {}'.format(precision))


def visualize(path):

    x, y_gt = dl.load_test_data(path)
    x_imput = x / 255.0 - 0.5
    m_net = MNISTNet()
    m_net.load_param()
    visualize_idx = random.randint(0, x.shape[0]-1)
    y_pred = m_net.predict(x_imput[visualize_idx])

    plt.subplot(111)
    title = "真值标签为：{},""预测标签为：{}".format(y_gt[visualize_idx], y_pred)
    plt.title(title, fontproperties='SimHei')
    img = np.array(x[visualize_idx]).reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
if __name__ == '__main__':
    # train_net()
    # eval_net()
    visualize("")