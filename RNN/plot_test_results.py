import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    exp_name = 'dann'
    PATH = PATH = './RNN/models/' + exp_name+ '/test_results/'
    
    fig,ax = plt.subplots(1, 2)
    
    source = pd.read_csv(PATH+'source/predicts.csv')
    sns.lineplot(data=source, x=range(len(source)), y='ICP', ax= ax[0], label='original')
    sns.lineplot(data=source, x=range(len(source)), y='dann_preds', ax= ax[0], label='test')
    ax[0].set_title('SOURCE')
    
    
    
    target = pd.read_csv(PATH+'target/predicts.csv')
    sns.lineplot(data=target, x=range(len(target)), y='ICP', ax= ax[1], label='original')
    sns.lineplot(data=target, x=range(len(target)), y='dann_preds', ax= ax[1], label='test')
    ax[1].set_title('TARGET')
    plt.legend()
    plt.show()
    
    