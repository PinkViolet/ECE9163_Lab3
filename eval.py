import keras
import sys
import h5py
import numpy as np
import argparse

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main(args):
    cl_x_test, cl_y_test = data_loader(args.val_data)
    bd_x_test, bd_y_test = data_loader(args.backdoor_data)
    print('clean valid dataset in range [%d, %d])' % (np.min(cl_y_test), np.max(cl_y_test)))
    print('backdoor valid dataset in range [%d, %d])' % (np.min(bd_y_test), np.max(bd_y_test)))

    bd_model = keras.models.load_model(args.bd_model)
    repair_model = keras.models.load_model(args.repair_model)

    cl_label_p = np.argmax(repair_model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print('Clean Classification accuracy:', clean_accuracy)
    
    bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
    repair_label_p = np.argmax(repair_model.predict(bd_x_test), axis=1)
    repair_label_p[bd_label_p != repair_label_p] = 1283
    asr = np.mean(np.equal(repair_label_p, bd_y_test))*100
    print('Attack Success Rate:', asr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prune the backdoor model.")
    parser.add_argument("--bd_model", type=str, default="models/bd_net.h5",
                        help="Path to model definition file (.h5)")
    parser.add_argument("--repair_model", type=str, default="models/repair_net_two_percent.h5",
                        help="Path to model definition file (.h5)")
    parser.add_argument('--val_data', type=str, default='data/cl/valid.h5',
                        help="Path to validation dataset file (.h5)")
    parser.add_argument('--backdoor_data', type=str, default='data/bd/bd_test.h5',
                        help="Path to backdoor dataset file (.h5)")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")
    main(args)
