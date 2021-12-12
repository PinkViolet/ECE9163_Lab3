import h5py
import numpy as np
import argparse
from keras.models import load_model
import keras
import matplotlib.pyplot as plt
# import tensorflow as tf
# from eval import data_loader
from model import Repaired_Model
# tf.enable_eager_execution()
# import torch
# torch.cuda.empty_cache()

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def evaluate(y_pred, y_truth):
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.mean(np.equal(y_pred, y_truth)) * 100
    return acc

def evaluate_backdoor_attack(y_bd, y_repair):
    y_bd = np.argmax(y_bd, axis=1)
    y_repair = np.argmax(y_repair, axis=1)
    atk_success = np.sum(y_bd == y_repair) / y_bd.shape[0] * 100
    return atk_success


# def check_backdoor(y):
#     y_bad, y_pred = y[0], y[1]
#     label_bad = keras.backend.argmax(y_bad, axis=1)
#     label_pred = keras.backend.argmax(y_pred, axis=1)
#     N_plus_1 = keras.backend.reshape(tf.cast((label_bad != label_pred), tf.int32), (-1, 1))
#     # y_pred[label_bad != label_pred] = np.zeros(1283)
#     # tf.scatter_update(y_pred, label_bad != label_pred, np.zeros(1283))
#     output_list = []
#     for i in range(y_pred.shape[0]):
#         if N_plus_1[i] == 1:
#             output_list.append(np.zeros(1283))
#         else:
#             output_list.append(y_pred[i])
#     y_pred = tf.stack(output_list)
#     y_pred = keras.backend.concatenate((y_pred, N_plus_1), axis=1)
#     return y_pred


def prune(args):
    badnet = load_model(args.model)
    repairnet = load_model(args.model)
    repairnet.summary()

    x_val, y_val = data_loader(args.val_data)
    x_test, y_test = data_loader(args.test_data)
    x_bd, y_bd = data_loader(args.backdoor_data)

    two_p = False
    four_p = False
    ten_p = False

    # fine_tuning
    # repairnet.fit(x_val, y_val, epochs=20)

    target_repair_rate = args.repair_rate
    current_repair_rate = 0.
    test_acc = [evaluate(repairnet.predict(x_test), y_test)]
    attack_success_rate = [evaluate_backdoor_attack(badnet.predict(x_bd), repairnet.predict(x_bd))]
    pruned_index = []

    while current_repair_rate < target_repair_rate:
        pool3 = keras.Model(inputs=repairnet.inputs,
                            outputs=repairnet.get_layer('pool_3').output)

        # find prune index
        activation = pool3.predict(x_val)
        activation = np.mean(np.sum(activation, axis=(1,2)), axis=0)
        pruned_c = np.argsort(activation)
        for c in pruned_c:
            if c not in pruned_index:
                pruned_c = c
                pruned_index.append(c)
                break

        # prune the repair network by zero out weights
        modified_weight = repairnet.layers[5].get_weights()
        modified_weight[0][..., pruned_c] = np.zeros_like(modified_weight[0][...,pruned_c])
        modified_weight[1][pruned_c] = 0
        repairnet.layers[5].set_weights(modified_weight)

        test_acc.append(evaluate(repairnet.predict(x_test), y_test))
        attack_success_rate.append(evaluate_backdoor_attack(badnet.predict(x_bd), repairnet.predict(x_bd)))
        current_repair_rate += 100/60

        if not two_p and test_acc[0]-test_acc[-1] >= 2:
            print('Save models at 2% of accuracy drop')
            repairnet.save('models/repair_net_two_percent.h5')
            two_p = True
        if not four_p and test_acc[0]-test_acc[-1] >= 4:
            print('Save models at 4% of accuracy drop')
            repairnet.save('models/repair_net_four_percent.h5')
            four_p = True
        if not ten_p and test_acc[0]-test_acc[-1] >= 10:
            print('Save models at 10% of accuracy drop')
            repairnet.save('models/repair_net_ten_percent.h5')
            ten_p = True

        # print(pruned_index, test_acc)

    print('finish pruning')
    print('generate plots')
    plt.plot(np.arange(len(test_acc))/60, test_acc, label='Accuracy on Clean Data')
    plt.plot(np.arange(len(test_acc))/60, attack_success_rate, label='Attack Success Rate on Backdoor Data')
    plt.xlabel('Prune Ratio')
    plt.ylabel('%')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prune the backdoor model.")
    parser.add_argument("-m", "--model", type=str, default="models/bd_net.h5",
                        help="Path to model definition file (.h5)")
    parser.add_argument('-w', '--weight', type=str, default='models/bd_weights.h5',
                        help="Path to model weight file (.h5)")
    parser.add_argument('--val_data', type=str, default='data/cl/valid.h5',
                        help="Path to validation dataset file (.h5)")
    parser.add_argument('--test_data', type=str, default='data/cl/test.h5',
                        help="Path to validation dataset file (.h5)")
    parser.add_argument('--backdoor_data', type=str, default='data/bd/bd_test.h5',
                        help="Path to backdoor dataset file (.h5)")
    parser.add_argument('--repair_rate', type=float, default=100,
                        help='Acceptable performance decay in % unit')
    args = parser.parse_args()
    print(f"Command line arguments: {args}")
    prune(args)