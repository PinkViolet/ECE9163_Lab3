import keras
from keras.layers import Input
from keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
from keras.models import load_model


class Repaired_Model(keras.layers.Layer):
    def __init__(self, badnet, repairnet):
        super(Repaired_Model, self).__init__()
        # BadNet
        # self.badnet = load_model(args.model)
        # self.badnet.load_weights(args.weight)
        # self.badnet.summary()

        # RepairNet
        # self.repairnet = load_model(args.model)
        # self.repairnet.load_weights(args.weight)
        # input = Input(shape=(55,47,3))
        # repairnet = repairnet(input)
        # badnet = badnet(input)
        # self.model = Model(inputs=input, outputs=[badnet, repairnet])
        self.badnet = badnet
        self.repairnet = repairnet
        self.class_num = self.badnet.output.shape[1]
        # self.model.summary()

        # self.pool3_output = Sequential([badnet.get_layer('input'),
        #                           badnet.get_layer('conv_1'),
        #                           badnet.get_layer('pool_1'),
        #                           badnet.get_layer('conv_2'),
        #                           badnet.get_layer('pool_2'),
        #                           badnet.get_layer('conv_3'),
        #                           badnet.get_layer('pool_3')])
        # self.branch1 = Sequential([badnet.get_layer('flatten_1'),
        #                             badnet.get_layer('fc_1')])
        # self.branch2 = Sequential([badnet.get_layer('conv_4'),
        #                            badnet.get_layer('flatten_2'),
        #                            badnet.get_layer('fc_2')])
        # self.head = Sequential([badnet.get_layer('activation_1'),
        #                         badnet.get_layer('output')])

    def call(self, x):
        y_bad = self.badnet.predict(x)
        print('y_bad')
        y_pred = self.repairnet.predict(x)
        print('y_repair')
        # y_bad, y_pred = self.model(x)

        label_bad = np.argmax(y_bad, axis=1)
        label_pred = np.argmax(y_pred, axis=1)
        N_plus_1 = (label_bad != label_pred).astype(int).reshape((-1,1))
        y_pred[label_bad != label_pred] = np.zeros(self.class_num)
        y_pred = np.hstack((y_pred, N_plus_1))
        print(y_pred.shape)
        # pool3_output = self.pool3_output(x)
        #
        # # pruning
        # for c in pruned_c:
        #     pool3_output[...,c] = 0
        #
        # y_repair = self.branch1(pool3_output) + self.branch2(pool3_output)
        # y_repair = self.head(y_repair)
        # y_repair = tf.argmax(y_repair, axis=1)
        #
        # mask = tf.math.not_equal(y_bad, y_repair)
        # y_repair = tf.scatter_update(y_repair, mask, tf.constant(-1))
        return y_pred
