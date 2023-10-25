import torch
import torch.nn as nn
import warnings
import numpy as np
import pickle
import random
import yaml
import os

# import our functions
from utils import aux_functions 
from utils import net_models
from utils import train_and_test_functions

def main(path_to_config=None, run_name=None):

    if path_to_config==None or run_name==None:
        warnings.warn('\nConfig and folder with its location need to be specified. Exiting.')
        exit()

    with open(path_to_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config['loss_function'] = nn.MSELoss(reduction='sum') 
    algorithm = config['algorithm']
    data = config['data']
    train_mode = config['train_mode']
    adversarial_attacks = config['adversarial_attacks']
    hidden_size = config['hidden_size']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    double_precision = config['double_precision']
    num_seeds = config['num_seeds']
    lr = config['lr']
    gamma = config['gamma']
    B_scaling = config['B_scaling']
    with_bias = config['with_bias']
    activation_function = config['activation_function']
    save_performances_values = config['save_performances_values']

    if double_precision:
        torch.set_default_dtype(torch.float64)

    # define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Code will use device: ' + device.upper())

    print('\nlr = {}'.format(config['lr']))
    
    # define network internal parameters
    if data=='MNIST' or data=='FMNIST':
        num_classes = 10
        input_size = 28*28
    elif data=='CIFAR10':
        num_classes = 10
        input_size = 32*32*3
    elif data=='CIFAR100':
        num_classes = 100
        input_size = 32*32*3
    else:
        warnings.warn('\nDataset not defined. Exiting.')
        exit()
    
    # create list of random seeds
    seeds_list = np.random.randint(0, 10000, num_seeds)
    
    # add sizes of data to config
    config['num_classes'] = num_classes
    config['input_size'] = input_size
    
    if save_performances_values:
        mode = algorithm + '_' + train_mode + '_' + data
        
        # create directories of interest
        results_out_dir = os.path.join(os.path.join('logs/results/', run_name), mode)
        net_models_out_dir = os.path.join(os.path.join('logs/net_models/', run_name), mode)
        dir_list = [results_out_dir, net_models_out_dir]
        for dir in dir_list:
            exist = os.path.exists(dir)
            if not exist:
                os.makedirs(dir)

        # instantiate placeholders for measures/variables to be saved
        train_loss_by_epoch_plot = np.zeros((len(seeds_list), num_epochs))
        train_acc_by_epoch_plot = np.zeros((len(seeds_list), num_epochs))
        test_loss_by_epoch_plot = np.zeros((len(seeds_list), num_epochs))
        test_acc_by_epoch_plot = np.zeros((len(seeds_list), num_epochs))
        test_loss_adv_by_epoch_plot = np.zeros((len(seeds_list), num_epochs, len(adversarial_attacks)))
        test_acc_adv_by_epoch_plot = np.zeros((len(seeds_list), num_epochs, len(adversarial_attacks)))
    
    if not save_performances_values:
        # if not saving values, run for 1 seed
        seeds_list = np.random.randint(0, 10000, 1)
        
    for s, seed_id in enumerate(seeds_list):
        
        print("\n")
        print("Starting to train with fixed random seed {}.".format(seed_id))

        # fix the random seed 
        torch.cuda.empty_cache()
        torch.manual_seed(seed_id)
        torch.cuda.manual_seed(seed_id)
        torch.cuda.manual_seed_all(seed_id)
        np.random.seed(int(seed_id))
        random.seed(int(seed_id))
        
        # load datasets
        train_loader, val_loader, test_loader = aux_functions.get_data_loaders(data, batch_size, val=False)

        # initialize the model
        model = net_models.net(input_size, hidden_size, num_classes, with_bias, activation_function=activation_function).to(device)

        B_matrix=None
        activation=None
        v_w_all=None
        optimizer=None

        # internal PEPITA specific network parameters for training
        if algorithm=='PEPITA':
            sd = np.sqrt(6/input_size)
            B_matrix = (torch.rand(input_size,num_classes)*2*sd-sd)*B_scaling
            B_matrix = B_matrix.to(device)

            # function to record activations: to compare the activations of the 2 forward passes
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            hook_list = []
            for name, layer in model.named_modules():
                h = layer.register_forward_hook(get_activation(name))
                hook_list.append(h)

            v_w_all = []
            for w in model.parameters():
                with torch.no_grad():
                    v_w_all.append(torch.zeros(w.shape).to(device))
        
        # instantiate learning rates
        lr = config['lr']
        if algorithm=='BP':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=gamma, weight_decay=0)
    
        # start training the model
        for e in range(num_epochs):

            # learning rate decay
            if e in [60,90]:
                lr = lr*0.1
                if algorithm=='BP':
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=gamma, weight_decay=0)

            acc_train, loss_train = train_and_test_functions.training_epoch(model, device, train_loader, config, e, lr, optimizer, B_matrix, activation, v_w_all, out=False)
            
            if save_performances_values:
                train_loss_by_epoch_plot[s,e] = loss_train
                train_acc_by_epoch_plot[s,e] = acc_train
                acc_test, loss_test = train_and_test_functions.testing_performance(model, device, test_loader, config)
                aux_functions.progress(len(test_loader), len(test_loader), 'Batch [{}/{}] Epoch [{}/{}]  Test loss = {:.4f}  Test acc = {:.3f}%'.format(len(test_loader), len(test_loader), e+1, num_epochs, loss_test, acc_test))
                acc_test_adv, loss_test_adv = train_and_test_functions.testing_adversarial_performance(model, device, test_loader, config, adversarial_attacks)
                test_loss_by_epoch_plot[s,e] = loss_test
                test_acc_by_epoch_plot[s,e] = acc_test
                test_loss_adv_by_epoch_plot[s,e] = loss_test_adv
                test_acc_adv_by_epoch_plot[s,e] = acc_test_adv

    print('Training finished.\n')
   
    # save parameters and variables of interest
    if save_performances_values:

        # saving the trained network
        if algorithm=='PEPITA':
            for h in hook_list:
                h.remove()
        file_name = 'network.pt'
        path = os.path.join(net_models_out_dir, file_name)
        torch.save(model, path) 

        # saving the performance results of the model 
        file_name = 'performances.pickle'
        path = os.path.join(results_out_dir, file_name)
        performance_results = [train_acc_by_epoch_plot, test_acc_by_epoch_plot, test_acc_adv_by_epoch_plot, train_loss_by_epoch_plot, test_loss_by_epoch_plot, test_loss_adv_by_epoch_plot]
        with open(path, 'wb') as handle:
            pickle.dump(performance_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        
        if 'table_2' in run_name:
            best_epoch = np.argmax(np.mean(test_acc_adv_by_epoch_plot[:,:,0],axis=0))
        else:
            best_epoch = np.argmax(np.mean(test_acc_by_epoch_plot,axis=0))
        
        print('Testing Accuracy (last, best adv): {:0.3f}%, {:.2f} / {:0.3f}%, {:.2f} (epoch={})'.format(np.mean(test_acc_by_epoch_plot,axis=0)[-1],
                                                                                                np.std(test_acc_by_epoch_plot,axis=0)[-1], 
                                                                                                np.mean(test_acc_by_epoch_plot,axis=0)[best_epoch], 
                                                                                                np.std(test_acc_by_epoch_plot,axis=0)[best_epoch],
                                                                                                best_epoch))
        for j in range(len(adversarial_attacks)):
            print('Adversarial Accuracy to {} attack (last, best adv): {:0.3f}%, {:.2f} / {:0.3f}%, {:.2f} (epoch={})'.format(adversarial_attacks[j], 
                                                                                                                            np.mean(test_acc_adv_by_epoch_plot[:,:,j],axis=0)[-1], 
                                                                                                                            np.std(test_acc_adv_by_epoch_plot[:,:,j],axis=0)[-1], 
                                                                                                                            np.mean(test_acc_adv_by_epoch_plot[:,:,j],axis=0)[best_epoch],
                                                                                                                            np.std(test_acc_adv_by_epoch_plot[:,:,j],axis=0)[best_epoch],
                                                                                                                            best_epoch))