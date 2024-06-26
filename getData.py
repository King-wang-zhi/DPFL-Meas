import numpy as np

class GetDataSetbloodmnist(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'bloodmnist':
            self.mnistDataSetConstruct(isIID)
        else:
            pass


    def mnistDataSetConstruct(self, isIID):
        # train bloodmnist
        bloodmnist_data = np.load('./data/bloodmnist.npz')

        train_images = bloodmnist_data['train_images']
        test_images = bloodmnist_data['test_images']
        train_labels = bloodmnist_data['train_labels']
        test_labels = bloodmnist_data['test_labels']

        #train bloodmnist
        train_images = train_images.reshape(11959, 28, 28, 3)
        test_images = test_images.reshape(3421, 28, 28, 3)


        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 3
        assert test_images.shape[3] == 3
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2] * train_images.shape[3])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2] * test_images.shape[3])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)   # 产生数组0到n-1
            np.random.shuffle(order)                  # 打乱数组
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # labels = np.argmax(train_labels, axis=1)  # 按行方向返回最大值索引
            labels = train_labels.reshape(-1)
            order = np.argsort(labels)                # 返回从小到大排序元素对应的索引
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        # 数据分配方式是否是独立同分布的



        self.test_data = test_images
        self.test_label = test_labels

class GetDataSetpathmnist(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'pathmnist':
            self.mnistDataSetConstruct(isIID)
        else:
            pass


    def mnistDataSetConstruct(self, isIID):
        # train pathmnist
        pathmnist_data = np.load('./data/pathmnist.npz')

        train_images = pathmnist_data['train_images']
        test_images = pathmnist_data['test_images']
        train_labels = pathmnist_data['train_labels']
        test_labels = pathmnist_data['test_labels']

        #train pathmnist
        train_images = train_images.reshape(89996, 28, 28, 3)
        test_images = test_images.reshape(7180, 28, 28, 3)


        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 3
        assert test_images.shape[3] == 3
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2] * train_images.shape[3])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2] * test_images.shape[3])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)   # 产生数组0到n-1
            np.random.shuffle(order)                  # 打乱数组
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # labels = np.argmax(train_labels, axis=1)  # 按行方向返回最大值索引
            labels = train_labels.reshape(-1)
            order = np.argsort(labels)                # 返回从小到大排序元素对应的索引
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        # 数据分配方式是否是独立同分布的



        self.test_data = test_images
        self.test_label = test_labels

class GetDataSetpneumoniamnist(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'pneumoniamnist':
            self.mnistDataSetConstruct(isIID)
        else:
            pass


    def mnistDataSetConstruct(self, isIID):
        # train pneumoniamnist
        pneumoniamnist_data = np.load('./data/pneumoniamnist.npz')

        train_images = pneumoniamnist_data['train_images']
        test_images = pneumoniamnist_data['test_images']
        train_labels = pneumoniamnist_data['train_labels']
        test_labels = pneumoniamnist_data['test_labels']

        #train pneumoniamnist
        train_images = train_images.reshape(4708, 28, 28, 1)
        test_images = test_images.reshape(624, 28, 28, 1)


        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2] * train_images.shape[3])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2] * test_images.shape[3])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)   # 产生数组0到n-1
            np.random.shuffle(order)                  # 打乱数组
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # labels = np.argmax(train_labels, axis=1)  # 按行方向返回最大值索引
            labels = train_labels.reshape(-1)
            order = np.argsort(labels)                # 返回从小到大排序元素对应的索引
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        # 数据分配方式是否是独立同分布的



        self.test_data = test_images
        self.test_label = test_labels

if __name__=="__main__":
    'test data set from bloodmnist'
    mnistDataSet = GetDataSetbloodmnist('bloodmnist', True) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))

