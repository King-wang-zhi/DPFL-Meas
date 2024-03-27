import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import random

from getData import *
from dp_mechanism import cal_sensitivity_pdp, cal_sensitivity_gdp, Laplace, Gaussian_Simple
from utils import config
from minimizers import ASAM

import shutil
import datetime
import time
import os.path
import sys

config = config()

class client(object):
    def __init__(self, trainDataSet, dev, dp_mechanism='no_dp', attack='no_attack', dp_epsilon=0, dp_delta=1e-5, dp_clip=0.2, idxs=None, rho=0.5, eta=0.01, lr=0.1):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.dp_mechanism = 'no_dp'
        self.dp_epsilon = dp_epsilon  # Privacy budget
        self.dp_delta = dp_delta  # Relaxation difference
        self.dp_clip = dp_clip  # Gradient cropping
        self.idxs = idxs
        self.lr = lr
        self.rate = random.random()
        self.turnlabel = 0
        self.attack = 'no_attack'
        self.rate = 0.5

        self.rho = rho
        self.eta = eta
        self.dp_process = ''

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, dp_mechanism, num_comm,
                    now_num_comm, clientnum, attack, model_name, asam, dp_process, privacy_budget, iid):

        now = str(datetime.datetime.now())[:19]
        now = now.replace(":", "_")
        now = now.replace("-", "_")
        now = now.replace(" ", "_")
        self.dp_process = dp_process
        self.dp_epsilon = privacy_budget

        src_dir = config.path.data_path
        path = "config/" + str(model_name) + "_" + str(config.statistics.type)
        if os.path.exists(path) == False:
            os.mkdir(path)
        dst_dir = path + "/config.yaml"
        shutil.copy(src_dir, dst_dir)
        self.attack = attack
        Net.load_state_dict(global_parameters, strict=True)
        data_train_target = self.train_ds

        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True, drop_last=True)

        Loss_list = []
        # Perform pretraining to calculate DP ratio
        start = time.time()
        for epoch in range(localEpoch):
            if asam == 1:
                minimizer = ASAM(opti, Net, self.rho, self.eta)
            running_loss = 0
            for data, label in self.train_dl:
                data = data.reshape(10, 3, 28, 28)
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()

                if asam == 1:
                    # Ascent Step
                    minimizer.ascent_step()
                    # Descent Step
                    lossFun(Net(data), label).backward()
                    if dp_mechanism != 'no_dp':
                        self.clip_gradients(Net, dp_mechanism)
                        self.add_noise(Net, dp_mechanism)
                    minimizer.descent_step()
                else:
                    if dp_mechanism != 'no_dp':
                        self.clip_gradients(Net, dp_mechanism)
                        self.add_noise(Net, dp_mechanism)
                    opti.step()
                    opti.zero_grad()
                running_loss += loss.item()

            Loss_list.append(running_loss)

        print("长度", len(self.train_ds))

        torch.save(Net.state_dict(), 'model.pth')

        if dp_mechanism != 'no_dp':
            self.add_noise(Net, dp_mechanism)

        end = time.time()
        times = end - start

        # Record time information
        file_path_time = 'results/time/{}/client/{}/client_{}_{}{}.txt'.format(
            'IID' if iid == 1 else 'NonIID',
            'Blood' if 'blood' in model_name else 'Path' if 'path' in model_name else 'Pneumonia',
            'cnn' if 'cnn' in model_name else 'ViT',
            'Gaussian' if dp_mechanism == 'Gaussian' else 'Laplace' if dp_mechanism == 'Laplace' else 'no_dp',
            '' if dp_mechanism == 'no_dp' else '_LDP' if dp_process == 'ldp' else '_GDP'
        )
        with open(file_path_time, 'a') as f:
            f.write('%d %.3f\n' % (now_num_comm, times))

        # Record transmission data size
        Transmission = sys.getsizeof(Net.state_dict())

        file_path_trans = 'results/transmission/{}/{}/{}_{}{}.txt'.format(
            'IID' if iid == 1 else 'NonIID',
            'Blood' if 'blood' in model_name else 'Path' if 'path' in model_name else 'Pneumonia',
            'cnn' if 'cnn' in model_name else 'ViT',
            'Gaussian' if dp_mechanism == 'Gaussian' else 'Laplace' if dp_mechanism == 'Laplace' else 'no_dp',
            '' if dp_mechanism == 'no_dp' else '_LDP' if dp_process == 'ldp' else '_GDP'
        )
        with open(file_path_trans, 'a') as f:
            f.write('%d %.3f\n' % (now_num_comm, Transmission))

        return Net.state_dict()

    def local_val(self):
        pass

    def clip_gradients(self, net, dp_mechanism):
        if dp_mechanism == 'Laplace':
            for k, v in net.named_parameters():
                torch.nn.utils.clip_grad_value_(v, self.dp_clip)

        elif dp_mechanism == 'Gaussian':
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(2) / self.dp_clip)


    def add_noise(self, net, dp_mechanism):
        if self.dp_process == 'pdp':
            sensitivity = cal_sensitivity_pdp(self.dp_clip, len(self.train_ds), self.lr)

        if self.dp_process == 'gdp':
            sensitivity = cal_sensitivity_gdp(self.dp_clip, len(self.train_ds))

        if dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    v += noise

        elif dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.dev)
                    v += noise


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.TestTensorDataset = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        if self.data_set_name == 'pneumoniamnist':
            mnistDataSet = GetDataSetpneumoniamnist(self.data_set_name, self.is_iid)

        elif self.data_set_name == 'bloodmnist':
            mnistDataSet = GetDataSetbloodmnist(self.data_set_name, self.is_iid)

        elif self.data_set_name == 'pathmnist':
            mnistDataSet = GetDataSetpathmnist(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        # test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)

        test_label = []
        for i in range(mnistDataSet.test_data_size):
            test_label.append(mnistDataSet.test_label[i][0])
        test_label = torch.tensor(test_label, dtype=torch.int64)

        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=10, shuffle=False,
                                           drop_last=True)
        self.test_data_loaders = DataLoader(TensorDataset(test_data, test_label), batch_size=600, shuffle=False,
                                            drop_last=True)
        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data = np.vstack((data_shards1, data_shards2))
            local_label = []
            for j in range(shard_size):
                local_label.append(label_shards1[j][0])
            for j in range(shard_size):
                local_label.append(label_shards2[j][0])
            # local_label = np.argmax(local_label, axis=1)
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label, dtype=torch.int64)),
                             self.dev)
            self.clients_set['client{}'.format(i)] = someone

if __name__=="__main__":
    MyClients = ClientsGroup('pathmnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])