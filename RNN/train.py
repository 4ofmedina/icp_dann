import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from networks import FlowCNN, FlowCNNRecons, RNN, LSTM, BiLSTM, Extractor, Domain_classifier
from dataset import BloodFlow, ToTensor
from tensorboardX import SummaryWriter
import numpy as np
import os

# parameters
N_STEPS = 1
N_INPUTS = 300
N_NEURONS = 512
N_OUTPUTS = 1
N_EPOCHS = 5
BATCH_SIZE = 64
N_LAYERS = 2
gamma = 10
theta = 0.1

if __name__ == '__main__':  

    # experiment - save model
    exp_name = 'dann'
    model_name = 'model.pt'
    PATH = './RNN/models/' + exp_name
    
    if not os.path.exists('./RNN/models/'):
                os.mkdir('./RNN/models/')
    

    # load training dataset
    source_trainset = BloodFlow(csv_file='flows-train.csv', root_dir='./RNN/data/source', transform=ToTensor())
    source_trainloader = DataLoader(source_trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)
    
    target_trainset = BloodFlow(csv_file='flows-train.csv', root_dir='./RNN/data/target', transform=ToTensor())
    target_trainloader = DataLoader(target_trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

    # load validation dataset
    source_valset = BloodFlow(csv_file='flows-val.csv', root_dir='./RNN/data/source', transform=ToTensor())
    source_valloader = DataLoader(source_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

    target_valset = BloodFlow(csv_file='flows-val.csv', root_dir='./RNN/data/target', transform=ToTensor())
    target_valloader = DataLoader(target_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model instance
    
    lstm = LSTM(input_size=N_INPUTS, output_size=N_OUTPUTS, hidden_dim=N_NEURONS, n_layers=N_LAYERS, device=device)
    regressor = Regressor()
    domain_classifier = Domain_classifier()
    domain_criterion = nn.NLLLoss()
    
    lstm = lstm.to(device)
    regressor = regressor.to(device)
    domain_classifier = domain_classifier.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params': lstm.parameters()},
                            {'params': domain_classifier.parameters()},
                            {'params': regressor.parameters()}], lr=1E-3)   #

    # Summary writers
    writer_train = SummaryWriter(PATH + '/train')
    writer_val = SummaryWriter(PATH + '/val')

    best_val_loss = np.inf
    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
        train_running_loss = 0.0
        val_running_loss = 0.0

        lstm.train()
        regressor.train()
        domain_classifier.train()
        
        # steps
        start_steps = epoch * len(source_trainloader)
        total_steps = N_EPOCHS * len(source_trainloader)

        # TRAINING ROUND
        for i, (sdata, tdata) in enumerate(zip(source_trainloader, target_trainloader)):
            
            # setup hyperparameters
            p = float(i + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-gamma * p)) - 1
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs
            input1, label1 = sdata
            input2, label2 = tdata
            size = min((input1.shape[0], input2.shape[0]))
            input1, label1 = input1[0:size], label1[0:size]
            input2, label2 = input2[0:size], label2[0:size]

            input1, label1 = input1.to(device), label1.to(device)
            input2, label2 = input2.to(device), label2.to(device)
            
            # prepare domain labels
            source_labels = torch.zeros((input1.size()[0])).type(torch.LongTensor).to(device)
            target_labels = torch.ones((input2.size()[0])).type(torch.LongTensor).to(device)
            
            # compute the lstm loss of src_feature
            preds, _ = lstm(input1)
            preds = preds.view(-1)
            lstm_loss = criterion(preds, label1)
            
            # compute the domain loss of src_feature and target_feature
            src_preds = domain_classifier(input1, constant)
            src_preds = src_preds.squeeze()
            tgt_preds = domain_classifier(input2, constant)
            tgt_preds = tgt_preds.squeeze()
            
            
            tgt_loss = domain_criterion(tgt_preds, target_labels)
            src_loss = domain_criterion(src_preds, source_labels)
            domain_loss = tgt_loss + src_loss
            
            loss = lstm_loss + theta * domain_loss
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly

            train_running_loss += loss.detach().item()

        lstm.eval()
        n_batches_train = np.copy(i)

        # VALIDATION ROUND
        for i, (sdata, tdata) in enumerate(zip(source_valloader, target_valloader)):

            # get the inputs
            input1, label1 = sdata
            input2, label2 = tdata
            size = min((input1.shape[0], input2.shape[0]))
            input1, label1 = input1[0:size], label1[0:size]
            input2, label2 = input2[0:size], label2[0:size]

            input1, label1 = input1.to(device), label1.to(device)
            input2, label2 = input2.to(device), label2.to(device)

            # forward
            preds, _ = lstm(input1)
            preds = preds.view(-1)
            val_loss = criterion(preds, label1)

            val_running_loss += val_loss.detach().item()

        n_batches_val = np.copy(i)

        
        
        print('Epoch:  {:3d} | LSTM Loss: {:.4f} | Domain Loss: {:.4f} | Val Loss: {:.4f} '.format(epoch,
                                                                      lstm_loss, domain_loss, val_running_loss / n_batches_val))
        
        
        # write summaries
        writer_train.add_scalar('loss', train_running_loss / n_batches_train, epoch)
        writer_val.add_scalar('loss', val_running_loss / n_batches_val, epoch)

        # save model at minimum validation loss
        if (val_running_loss / i) < best_val_loss:
            print('saving model')
            best_val_loss = val_running_loss / n_batches_val
            # save model
            if not os.path.exists(PATH):
                os.mkdir(PATH)
            torch.save(lstm.state_dict(), os.path.join(PATH, model_name))

    print('Optimization finished!')

