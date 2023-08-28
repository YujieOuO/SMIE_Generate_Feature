import os
from sacred import Experiment

ex = Experiment("Generate Features", save_git_info=False) 

@ex.config
def my_config():
    ############################## setting ##############################
    dataset = "pku"
    split = "split_9"
    epoch = 150
    lr = 5e-3
    weight_decay = 0
    hidden_size = 256
    ############################## ST-GCN ###############################
    in_channels = 3
    hidden_channels = 16
    hidden_dim = 256
    dropout = 0.5
    graph_args = {
    "layout" : 'ntu-rgb+d',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    ############################ downstream ############################
    batch_size = 128
    channel_num = 3
    person_num = 2
    joint_num = 25
    max_frame = 50
    train_list = "/home/u2020101280/ZeroShotSkeleton/data/zeroshot/"+dataset+"/"+split+"/seen_train_data.npy"
    train_label = "/home/u2020101280/ZeroShotSkeleton/data/zeroshot/"+dataset+"/"+split+"/seen_train_label.npy"
    test_list = "/home/u2020101280/ZeroShotSkeleton/data/zeroshot/"+dataset+"/"+split+"/seen_test_data.npy"
    test_label = "/home/u2020101280/ZeroShotSkeleton/data/zeroshot/"+dataset+"/"+split+"/seen_test_label.npy"
    ############################ save path ############################
    weight_path = "./model/"+split+".pt"
    log_path = "./log/"+split+".log"