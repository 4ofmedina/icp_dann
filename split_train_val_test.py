import os
import pandas as pd
from sklearn.model_selection import train_test_split


# directories
root_dir = os.getcwd()
source_csv_file = 'source_flows-stride-600.csv'
target_csv_file = 'target_flows-stride-600.csv'

if not os.path.exists('./RNN/data/'):
    os.mkdir('./RNN/data/')
if not os.path.exists('./RNN/data/source'):
    os.mkdir('./RNN/data/source')
if not os.path.exists('./RNN/data/target'):
    os.mkdir('./RNN/data/target')


##SOURCE
# read dataframe with all patients and windows
df = pd.read_csv(os.path.join(root_dir, source_csv_file))

# split train & test
dftrain, dftest = train_test_split(df, test_size=0.2, random_state=0)

dftrain, dfval = train_test_split(dftrain, test_size=0.125, random_state=0)

print(dftrain.shape)
print(dftest.shape)
print(dfval.shape)

# save csv files
dftrain.to_csv('./RNN/data/source/flows-train.csv')
dfval.to_csv('./RNN/data/source/flows-val.csv')
dftest.to_csv('./RNN/data/source/flows-test.csv')


##TARGET
# read dataframe with all patients and windows
df = pd.read_csv(os.path.join(root_dir, target_csv_file))

# split train & test
dftrain, dftest = train_test_split(df, test_size=0.2, random_state=0)

dftrain, dfval = train_test_split(dftrain, test_size=0.125, random_state=0)

# save csv files
dftrain.to_csv('./RNN/data/target/flows-train.csv')
dfval.to_csv('./RNN/data/target/flows-val.csv')
dftest.to_csv('./RNN/data/target/flows-test.csv')