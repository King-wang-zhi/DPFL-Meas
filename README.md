# DPFL-Meas

### environment
##### pytorch-version

1.python3.7.15

2.torch1.12.1+cu113

### datasets

The pneumoniamnist dataset are placed in the "data" folder called pneumoniamnist.npz.
Other datasets need to be downloaded from the website [(https://zenodo.org/records/10519652)](https://zenodo.org/records/10519652). They are "bloodmnist.npz" and "pathmnist.npz".

### usage

Run the code

```asp
python server.py -nc 100 -cf 0.1 -E 5 -B 10 -mn mnist_cnn  -ncomm 1000 -iid 0 -lr 0.01 -vf 20 -g 0

python server.py -nc 100 -cf 0.1 -E 4 -B 10 -mn mnist_2nn  -ncomm 50 -iid 1 -lr 0.01 -vf 1 -g 0 -dpm no_dp

python server.py -nc 100 -cf 0.2 -E 4 -B 10 -mn mnist_2nn  -ncomm 100 -iid 0 -lr 0.01 -vf 2 -g 0
```

```asp

python server.py -nc 1 -cf 1 -E 4 -B 10 -mn mnist_2nn  -ncomm 5 -iid 1 -lr 0.01 -vf 1 -g 0 -dpm no_dp
```

python server.py -nc 100 -cf 0.1 -E 4 -B 1 -mn pathmnist_2nn  -ncomm 10 -iid 1 -lr 0.01 -vf 1 -g 0 -dpm no_dp

which means there are 100 clients,  we randomly select 10 in each communicating round.  The data set are allocated in Non-IID way.  The epoch and batch size are set to 5 and 10. The learning rate is 0.01, we validate the codes every 20 rounds during the training, training stops after 1000 rounds. 

### Contributors.
This code was written by King-wang-zhi and organized by Zixuan He.
