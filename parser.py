import numpy as np

#   This class is used to parse the data from the Monk's dataset
class Monks_parser:
    
    def __init__(self, path_tr, path_ts):
        self.path_tr = path_tr
        self.path_ts = path_ts
        
    #   This function is used to parse the training data
    #   @param dim_features: number of features
    #   @param dim_out: number of labels
    #   @param one_hot: one hot encoding
    #   @param shuffle: True if you want shuffle data,
    #                   False otherwise
    #   @return: (X_train, Y_train, X_test, Y_test)
    def parse(self, dim_features, dim_out, one_hot=None, shuffle=False):
        dataset_train = self.__parse_file(self.path_tr, dim_features, dim_out, one_hot, shuffle)
        n_samples_train = dataset_train.shape[0]
        dataset_test = self.__parse_file(self.path_ts, dim_features, dim_out, one_hot, shuffle)
        n_samples_test = dataset_test.shape[0]
        if one_hot is not None:
            dim_features = one_hot

        X_train = dataset_train[:, dim_out:dim_features+dim_out].reshape((n_samples_train, dim_features))
        Y_train = dataset_train[:, 0].reshape((n_samples_train, dim_out))
        X_test = dataset_test[:, dim_out:dim_features+dim_out].reshape((n_samples_test, dim_features))
        Y_test = dataset_test[:, 0].reshape((n_samples_test, dim_out))
        return X_train, Y_train, X_test, Y_test

    #   This function is used to parse the file
    #   Shuffle with numpy.random.shuffle
    #   In this function I implemented the one hot encoding
    #   @param path: path of the file containing the dataset
    #   @param dim_features: number of features
    #   @param dim_out: number of labels
    #   @param one_hot: one hot encoding
    #   @param shuffle: True if you want shuffle data,
    #                   False otherwise
    #   @return: dataset
    ###
    def __parse_file(self, path, dim_features, dim_out, one_hot, shuffle):
        
        with open(path, 'r') as file:
            lines = file.readlines()
            if one_hot is not None:
                data = np.zeros((len(lines), one_hot + dim_out))
            else:
                data = np.zeros((len(lines), dim_features + dim_out))
            i = 0
            for line in lines:
                line = line.strip().split(' ')
                if one_hot is None:
                    data[i] = np.array(line[0:dim_features + dim_out])
                else:
                    #   !!one hot encoding implementation!!
                    data[i, 0:dim_out] = line[0:dim_out]
                    data[i, int(line[1])] = 1
                    data[i, int(line[2]) + 3] = 1
                    data[i, int(line[3]) + 6] = 1
                    data[i, int(line[4]) + 8] = 1
                    data[i, int(line[5]) + 11] = 1
                    data[i, int(line[6]) + 15] = 1
                i += 1
            file.close()
        if shuffle:
            np.random.shuffle(data)
        return data


