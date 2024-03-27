import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim
import time

from Models import PneumoniaMnist_CNN, BloodMnist_CNN,  PathMnist_CNN, ViT
from clients import ClientsGroup, client
from dp_mechanism import *

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

model_params = {
    'pneumoniamnist_ViT': (ViT, 'pneumoniamnist', 28, 4, 2, 1, 64, 6, 8, 128),
    'pneumoniamnist_cnn': (PneumoniaMnist_CNN, 'pneumoniamnist'),
    'bloodmnist_ViT': (ViT, 'bloodmnist', 28, 4, 8, 3, 64, 6, 8, 128),
    'bloodmnist_cnn': (BloodMnist_CNN, 'bloodmnist'),
    'pathmnist_ViT': (ViT, 'pathmnist', 28, 7, 9, 3, 64, 6, 8, 128),
    'pathmnist_cnn': (PathMnist_CNN, 'pathmnist')
}


def write_results(iid, dataset_name, result_type, dp_mechanism, dp_process, i, sum_accu,
                  num, running_loss, is_end):
    base_path_acc = f'results/accuracy/{iid}/{dataset_name}/'
    base_path_loss = f'results/loss/{iid}/{dataset_name}/'

    file_path_acc = f'{base_path_acc}{result_type}_FedSAM_{dp_mechanism}{dp_process}.txt'
    with open(file_path_acc, 'a') as f:
        f.write('%d %.3f\n' % (i + 1, (sum_accu.cpu().numpy()) / num))
        print('*************************************')

    file_path_loss = f'{base_path_loss}{result_type}_FedSAM_{dp_mechanism}{dp_process}.txt'
    with open(file_path_loss, 'a') as f:
        f.write('%d %.3f\n' % (i + 1, running_loss / num))

    if is_end:
        print("accuracy已保存至：", file_path_acc)
        print("loss已保存至：", file_path_loss)

def parse_opt():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=1, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='bloodmnist_cnn', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')
    parser.add_argument('-dpm', '--dp_mechanism', type=str, default='no_dp', help='run what kind of differential privarcy')
    parser.add_argument('-ak', '--attack_mechanism', type=str, default='no_attack', help='run what kind of attack')
    parser.add_argument('-asam', '--Fedasam', type=int, default=1, help='whether to use ASAM')
    parser.add_argument('-dpp', '--dp_process', type=str, default='pdp', help='parameter dp or gradient dp')
    opt = parser.parse_args()
    return opt

def federated_learning_training(opt):
    args = opt.__dict__
    privacy_budget = [10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    budget = 0
    start = time.time()

    Correct_list = []  # Establish accuracy in saving arrays
    Loss_list = []  # Establish an array to save loss
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create model objects and client groups
    if args['model_name'] in model_params:
        model_class, dataset_name, *params = model_params[args['model_name']]
        if len(params) > 0:
            net = model_class(image_size=params[0], patch_size=params[1], num_classes=params[2],
                              channels=params[3], dim=params[4], depth=params[5], heads=params[6],
                              mlp_dim=params[7])
        else:
            net = model_class()
        myClients = ClientsGroup(dataset_name, args['IID'], args['num_of_clients'], dev)
    else:
        print("Unknown model name:", args['model_name'])

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # Communication begins
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        attack = 'no_attack'
        sum_parameters = None
        The_first_client = 0
        time3 = time.time()
        for client in tqdm(clients_in_comm):
            if The_first_client == 0:
                if args['attack_mechanism'] == 'MIA':
                    if i+1 == 2 or i+1 == 4 or i+1 == 8 or i+1 == 16 or i+1 == 32:
                        attack = 'MIA'
                    The_first_client = 1
                if args['attack_mechanism'] == 'iDLG':
                    attack = 'iDLG'
                    The_first_client = 1
            else:
                attack = 'no_attack'
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters, args['dp_mechanism'], args['num_comm'], i + 1, client, attack, args['model_name'], args['Fedasam'], args['dp_process'], privacy_budget[budget],args['IID'])
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)
        time4 = time.time()
        times = time4 - time3

        # Record time information
        file_path = 'results/time/{}/server/{}/server_{}_{}.txt'.format(
            'IID' if args['IID'] == 1 else 'NonIID',
            'Blood' if 'blood' in args['model_name'] else 'Path' if 'path' in args['model_name'] else 'Pneumonia',
            'cnn' if 'cnn' in args['model_name'] else 'ViT',
            'Gaussian' if args['dp_mechanism'] == 'Gaussian' else 'Laplace' if args['dp_mechanism'] == 'Laplace' else 'no_dp'
            '' if args['dp_mechanism'] == 'no_dp' else '_LDP' if args['dp_process'] == 'ldp' else '_GDP'
        )
        with open(file_path, 'a') as f:
            f.write('%d %.3f\n' % (i + 1, times))

        # Model validation and saving model parameters
        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                running_loss = 0.0
                loss_list = []
                num = 0
                for data, label in testDataLoader:
                    data = data.reshape(10, 3, 28, 28)
                    # print(data.size())
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    loss = loss_func(preds, label)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                    running_loss += loss.item()
                    loss_list.append(loss.item())
                print('accuracy: {}'.format(sum_accu / num))
                Correct_list.append(100. * ((sum_accu.cpu().numpy()) / num))
                Loss_list.append(running_loss / num)

                # Record accuracy and loss
                write_results(  'IID' if args['IID'] == 1 else 'NonIID',
                                'Blood' if 'blood' in args['model_name'] else 'Path' if 'pathmnist' in args['model_name'] else 'Pneumonia',
                                'cnn' if 'cnn' in args['model_name'] else 'ViT',
                                'Gaussian' if args['dp_mechanism'] == 'Gaussian' else 'Laplace' if args['dp_mechanism'] == 'Laplace' else 'no_dp',
                                '' if args['dp_mechanism'] == 'no_dp' else '_LDP' if args['dp_process'] == 'ldp' else '_GDP',
                                i, sum_accu, num, running_loss, True if i == args['num_comm'] - 1 else False)

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
            '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                    i, args['epoch'],
                                                                    args['batchsize'],
                                                                    args['learning_rate'],
                                                                    args['num_of_clients'],
                                                                    args['cfraction'])))

    end = time.time()
    print('Time:', end - start)

if __name__=="__main__":
    args = parse_opt()
    federated_learning_training(args)