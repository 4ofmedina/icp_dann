import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BloodFlow, ToTensor
from networks import FlowCNN, FlowCNNRecons, RNN, LSTM, GRU
import numpy as np
import os
import pandas as pd

# parameters
N_STEPS = 1
N_INPUTS = 300
N_NEURONS = 512
N_OUTPUTS = 1
BATCH_SIZE = 64
N_LAYERS = 2


def test(testloader, testset):
    outs = []

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):

            # get the inputs and initialize hidden
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.hidden = model.init_hidden(inputs.shape[0])

            preds, _ = model(inputs)
            preds = preds.view(-1).cpu()

            outs.append(preds.detach().numpy())

    predictions = np.concatenate(outs, axis=0)
    n_test = predictions.shape[0]
    labels = testset.dataframe.values[:n_test, 5]

    error = np.abs(labels-predictions)
    print('\nTest Error: {:.2f}'.format(np.mean(error)))

    dftest = testset.dataframe
    dftest['dann_preds'] = predictions.tolist()
    dftest['errors'] = error.tolist()

    error_patients = []
    for patient_id in dftest['Patient ID'].unique():
        error_pat = dftest[dftest['Patient ID'] == patient_id]['errors'].values
        error_patients.append(np.mean(error_pat))
        print('Patient {}, Test Error: {:.2f}'.format(patient_id, np.mean(error_pat)))
        
    return dftest['dann_preds']


if __name__ == '__main__':

    # experiment - save model
    exp_name = 'dann'
    PATH = './RNN/models/' + exp_name + '/model.pt'

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model instance

    model = LSTM(input_size=N_INPUTS, output_size=N_OUTPUTS, hidden_dim=N_NEURONS, n_layers=N_LAYERS, device=device)
    model = model.to(device)

    model.load_state_dict(torch.load(PATH))
    model.eval()


    # load training dataset
    testset_source = BloodFlow(csv_file='flows-test.csv', root_dir='./RNN/data/source', transform=ToTensor())
    testset_target = BloodFlow(csv_file='flows-test.csv', root_dir='./RNN/data/target', transform=ToTensor())

    testloader_source = DataLoader(testset_source, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)
    testloader_target = DataLoader(testset_target, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)
    
    PATH = './RNN/models/'+exp_name+'/test_results/'
    if not os.path.exists(PATH):
                os.mkdir(PATH)
    if not os.path.exists(PATH+'source/'):
                os.mkdir(PATH+'source/')   
    if not os.path.exists(PATH+'target/'):
                os.mkdir(PATH+'target/')
            
    col_list = ["Patient ID", "Read", "ICP"]
    
    source = pd.read_csv('./RNN/data/source/flows-test.csv', usecols=col_list)
    df = test(testloader_source, testset_source)
    source = pd.concat([source,df], axis=1)
    with open(PATH+'source/predicts.csv', 'w') as f:
        source.to_csv(f, index=False)
    
    target = source = pd.read_csv('./RNN/data/target/flows-test.csv', usecols=col_list)
    df = test(testloader_target, testset_target)
    target = pd.concat([target,df], axis=1)
    with open(PATH+'target/predicts.csv', 'w') as f:
        target.to_csv(f, index=False)


        
        





