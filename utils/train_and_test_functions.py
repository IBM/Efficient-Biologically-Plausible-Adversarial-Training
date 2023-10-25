import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import advertorch.attacks as attacks
from utils import aux_functions

# Training epoch of the current model 
def training_epoch(model, device, loader, config, epoch, lr, optimizer, B_matrix, activation, v_w_all, out=True):
    
    # extract needed config parameters
    num_classes = config['num_classes']
    input_size = config['input_size']
    drop_rate = config['drop_rate']
    num_epochs = config['num_epochs']
    loss_function = config['loss_function']
    gamma = config['gamma']
    algorithm = config['algorithm']
    train_mode = config['train_mode']
    with_bias = config['with_bias']
    noisy_gradients = config['noisy_gradients']
    mean_noisy_gradients = config['mean_noisy_gradients']
    stddev_noisy_gradients = config['stddev_noisy_gradients']
    training_adversarial_attack = config['training_adversarial_attack']
    adv_eps = config['adv_eps']
    
    # keep track of score
    running_loss = 0
    score = 0

    # for each batch
    for j, (images, labels) in enumerate(loader):

        # process mini batch
        images, labels = images.to(device), labels.to(device)
        target_onehot = F.one_hot(labels, num_classes=num_classes).type(torch.DoubleTensor).to(device)
        images = images.view(-1, input_size)
        
        # generate adversarial samples
        model.eval()
        if training_adversarial_attack=='pgd':
            attack_images = attacks.LinfPGDAttack(model, loss_function, eps=adv_eps, clip_min=-1.0, clip_max=1.0)
        elif training_adversarial_attack=='fgsm':
            attack_images = attacks.GradientSignAttack(model, loss_function, eps=adv_eps, clip_min=-1.0, clip_max=1.0)
        adv_images = attack_images.perturb(images, target_onehot)
        model.train()
        
        # create dropout mask for all forward passes (same mask for the 2 passes of PEPITA)
        do_masks = []
        layers_for_mask = [l.shape[0] for l in model.parameters() if len(l.shape)>1]
        keep_rate = 1-drop_rate
        if drop_rate < 1:
            for l in layers_for_mask[:-1]:
                do_mask = Variable(torch.ones(len(images),l).bernoulli_(keep_rate, generator=torch.Generator(device='cpu')))/keep_rate
                do_masks.append(do_mask.to(device))
            # no dropout in the last layer
            do_masks.append(1)
        
        # BP model training
        if algorithm=='BP':
            # set parameters of all optimized tensors to zero
            optimizer.zero_grad()
            output = model(images, do_masks)
            if 'CrossEntropyLoss' in str(loss_function):
                loss = loss_function(output, labels) / len(images)
            elif 'MSELoss' in str(loss_function):
                loss = loss_function(output, target_onehot) / len(images)
            loss.backward(retain_graph=True)
            if noisy_gradients:
                model.fc1.weight.grad = model.fc1.weight.grad + (torch.randn_like(model.fc1.weight.grad) * stddev_noisy_gradients + mean_noisy_gradients)
                model.fc2.weight.grad = model.fc2.weight.grad + (torch.randn_like(model.fc2.weight.grad) * stddev_noisy_gradients + mean_noisy_gradients)
            optimizer.step()

        if algorithm=='BP' and train_mode=='adversarial':
            # set parameters of all optimized tensors to zero
            optimizer.zero_grad()
            output = model(adv_images, do_masks)
            if 'CrossEntropyLoss' in str(loss_function):
                loss = loss_function(output, labels) / len(images)
            elif 'MSELoss' in str(loss_function):
                loss = loss_function(output, target_onehot) / len(images)
            loss.backward()
            if noisy_gradients:
                model.fc1.weight.grad = model.fc1.weight.grad + (torch.randn_like(model.fc1.weight.grad) * stddev_noisy_gradients + mean_noisy_gradients)
                model.fc2.weight.grad = model.fc2.weight.grad + (torch.randn_like(model.fc2.weight.grad) * stddev_noisy_gradients + mean_noisy_gradients)
            optimizer.step()

        # PEPITA model training
        if algorithm=='PEPITA':
                        
            # 1st forward pass --> keep track of free activations
            output = model(images, do_masks)

            layers_act = []
            for i, key in enumerate(activation):
                if 'fc' in key:
                    # register the activations taking into account nonlinearities and dropout
                    layers_act.append(F.relu(activation[key]) * do_masks[i])
                    
            # compute the error
            error = output - target_onehot  

            # compute the modulated images
            error_input = error @ B_matrix.T
            # modify the input with the error
            mod_images = images + error_input
            
            # 2nd forward pass with modulated input --> keep track of modulated activations
            mod_output = model(mod_images, do_masks)
            mod_layers_act = []
            for i, key in enumerate(activation):
                if 'fc' in key:
                    # register the activations taking into account nonlinearities
                    mod_layers_act.append(F.relu(activation[key]) * do_masks[i])
            
            # compute the modulated error
            mod_error = mod_output - target_onehot
            
            # compute the delta_w for the batch
            delta_w_all = []
            for l in range(len(layers_act)):
                # update for the last layer
                if l == len(layers_act)-1:
                    # last layer seems to use the modulated error (not the error as the paper indicates)
                    if len(layers_act)>1:
                        delta_w = -mod_error.T @ mod_layers_act[-2]
                        if with_bias:
                            delta_b = -torch.sum(mod_error.T, dim=1)
                    else:
                        delta_w = -mod_error.T @ mod_images
                        if with_bias:
                            delta_b = -torch.sum(mod_error.T, dim=1)
                # update for the first layer
                elif l == 0:
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_images
                    if with_bias:
                        delta_b = -torch.sum((layers_act[l] - mod_layers_act[l]).T, dim=1)
                # update for the hidden layers (not first, not last)
                elif l>0 and l<len(layers_act)-1:
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_layers_act[l-1]
                    if with_bias:
                        delta_b = -torch.sum((layers_act[l] - mod_layers_act[l]).T, dim=1)
                delta_w_all.append(delta_w/len(images))
                if with_bias:
                    delta_w_all.append(delta_b/len(images))
                    
            for l_idx, w in enumerate(model.parameters()):
                with torch.no_grad():
                    v_w_all[l_idx] = gamma * v_w_all[l_idx] + lr * delta_w_all[l_idx]
                    w += v_w_all[l_idx]

        if algorithm=='PEPITA' and train_mode=='adversarial':
            
            # 1st forward pass --> keep track of free activations
            output = model(adv_images, do_masks)

            layers_act = []
            for i, key in enumerate(activation):
                if 'fc' in key:
                    # register the activations taking into account nonlinearities and dropout
                    layers_act.append(F.relu(activation[key]) * do_masks[i])
                    
            # compute the error
            error = output - target_onehot  

            # compute the modulated adv_images
            error_input = error @ B_matrix.T
            # modify the input with the error
            mod_adv_images = adv_images + error_input
            
            # 2nd forward pass with modulated input --> keep track of modulated activations
            mod_output = model(mod_adv_images, do_masks)
            mod_layers_act = []
            for i, key in enumerate(activation):
                if 'fc' in key:
                    # register the activations taking into account nonlinearities
                    mod_layers_act.append(F.relu(activation[key]) * do_masks[i])

            # compute the modulated error
            mod_error = mod_output - target_onehot

            # compute the delta_w for the batch
            delta_w_all = []
            for l in range(len(layers_act)):
                # update for the last layer
                if l == len(layers_act)-1:
                    # last layer seems to use the modulated error (not the error as the paper indicates)
                    if len(layers_act)>1:
                        delta_w = -mod_error.T @ mod_layers_act[-2]
                        if with_bias:
                            delta_b = -torch.sum(mod_error.T, dim=1)
                    else:
                        delta_w = -mod_error.T @ mod_images
                        if with_bias:
                            delta_b = -torch.sum(mod_error.T, dim=1)
                # update for the first layer
                elif l == 0:
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_images
                    if with_bias:
                        delta_b = -torch.sum((layers_act[l] - mod_layers_act[l]).T, dim=1)
                # update for the hidden layers (not first, not last)
                elif l>0 and l<len(layers_act)-1:
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_layers_act[l-1]
                    if with_bias:
                        delta_b = -torch.sum((layers_act[l] - mod_layers_act[l]).T, dim=1)
                delta_w_all.append(delta_w/len(images))
                if with_bias:
                    delta_w_all.append(delta_b/len(images))

            for l_idx, w in enumerate(model.parameters()):
                with torch.no_grad():
                    v_w_all[l_idx] = gamma * v_w_all[l_idx] + lr * delta_w_all[l_idx]
                    w += v_w_all[l_idx]

        # evaluation mode
        model.eval()
        # keep track of accuracy and loss
        if j==len(loader)-2:
            output = model(images, do_masks=None)
        else:
            output = model(images, do_masks=None)
        if 'CrossEntropyLoss' in str(loss_function):
            running_loss += loss_function(output, labels).item() / len(images)
        elif 'MSELoss' in str(loss_function):
            running_loss += loss_function(output, target_onehot).item() / len(images)
        inference = torch.argmax(output, axis=1)
        score += torch.sum(torch.eq(inference, labels)) / len(images)

        batch_accuracy = score.item() / (j+1) * 100
        batch_loss = running_loss / (j+1)
        if out and j % 8 == 0:
            aux_functions.progress(j+1, len(loader), 'Batch [{}/{}] Epoch [{}/{}] Train loss = {:.4f} Train acc = {:.3f}%'.format(j+1, len(loader), epoch+1, num_epochs, batch_loss, batch_accuracy))
        # train mode
        model.train()

    epoch_accuracy = score.item() / len(loader) * 100
    epoch_loss = running_loss / len(loader)

    return epoch_accuracy, epoch_loss



# Evaluating the testing performance of the current model 
def testing_performance(model, device, loader, config, adv_val=False, natural_trained_model=None):
    
    input_size = config['input_size']
    loss_function = config['loss_function']
    num_classes = config['num_classes']
    training_adversarial_attack = config['training_adversarial_attack']
    adv_eps = config['adv_eps']

    # evaluate the model
    model.eval()

    score = 0
    loss = 0

    if natural_trained_model is not None:
        model_to_create_images = natural_trained_model
    else:
        model_to_create_images = model
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(-1, input_size) 
        target_onehot = F.one_hot(labels, num_classes=num_classes).type(torch.DoubleTensor).to(device)

        if adv_val:
            # create adversarial samples
            if training_adversarial_attack=='pgd':
                attack_images = attacks.LinfPGDAttack(model_to_create_images, loss_function, eps=adv_eps, clip_min=-1.0, clip_max=1.0)
            elif training_adversarial_attack=='fgsm':
                attack_images = attacks.GradientSignAttack(model_to_create_images, loss_function, eps=adv_eps, clip_min=-1.0, clip_max=1.0)
            adv_images = attack_images.perturb(images, target_onehot)
            # augment the dataset with the adversarial samples
            images = torch.cat([images, adv_images], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            target_onehot = torch.cat([target_onehot, target_onehot], dim=0)
        output = model(images, do_masks=None)
        if 'CrossEntropyLoss' in str(loss_function):
            loss += loss_function(output, labels).item() / len(images)
        elif 'MSELoss' in str(loss_function):
            loss += loss_function(output, target_onehot).item() / len(images)
        inference = torch.argmax(output, axis=1)
        score += torch.sum(torch.eq(inference, labels)) / len(images)
    
    accuracy = score.item() / len(loader) * 100
    loss = loss / len(loader)

    return accuracy, loss



# Evaluating the adversarial performance of the current model 
def testing_adversarial_performance(model, device, loader, config, adversarial_attacks, natural_trained_model=None):
    
    input_size = config['input_size']
    loss_function = config['loss_function']
    num_classes = config['num_classes']
    adv_eps = config['adv_eps']

    mean_adv_acc_list = np.zeros(len(adversarial_attacks))
    mean_adv_loss_list = np.zeros(len(adversarial_attacks))

    # evaluate the model
    model.eval()

    if natural_trained_model != None:
        model_to_create_images = natural_trained_model
    else:
        model_to_create_images = model
    
    for n, attack in enumerate(adversarial_attacks):
        
        # define adversarial attack
        if attack=='pgd':
            adversary = attacks.LinfPGDAttack(model_to_create_images, loss_function, eps=adv_eps, clip_min=-1.0, clip_max=1.0)
        elif attack=='fgsm':
            adversary = attacks.GradientSignAttack(model_to_create_images, loss_function, eps=adv_eps, clip_min=-1.0, clip_max=1.0)
        elif attack=='bim':
            adversary = attacks.LinfBasicIterativeAttack(model_to_create_images, loss_function, eps=adv_eps, nb_iter=10, clip_min=-1.0, clip_max=1.0)
        elif attack=='mi-fgsm':
            adversary = attacks.MomentumIterativeAttack(model_to_create_images, loss_function, eps=adv_eps, nb_iter=10, decay_factor=0.8, clip_min=-1.0, clip_max=1.0)
        else:
            print('Adversarial attack not implemented. Exiting.')
            exit()

        score = 0
        loss = 0   
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, input_size) 
            target_onehot = F.one_hot(labels, num_classes=num_classes).type(torch.DoubleTensor).to(device)
            adv_images = adversary.perturb(images, target_onehot)
            output = model(adv_images, do_masks=None)
            if 'CrossEntropyLoss' in str(loss_function):
                loss += loss_function(output, labels).item() / len(images)
            elif 'MSELoss' in str(loss_function):
                loss += loss_function(output, target_onehot).item() / len(images)
            inference = torch.argmax(output, axis=1)
            score += torch.sum(torch.eq(inference, labels)) / len(images)
        accuracy = score.item() / len(loader) * 100
        loss = loss / len(loader)
        mean_adv_acc_list[n] = accuracy
        mean_adv_loss_list[n] = loss
    
    return mean_adv_acc_list, mean_adv_loss_list