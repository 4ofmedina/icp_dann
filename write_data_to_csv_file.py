import os
import csv
import numpy as np

source_datadir = "./results/source"
target_datadir = "./results/target"

# create CSV file
# image_name # flow # ICP value
source_filename = 'source_flows-stride-600.csv'
target_filename = 'target_flows-stride-600.csv'

def data_to_csv(datadir, filename):
    with open(filename, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # list all files in directory
        files = sorted(os.listdir(datadir))
        if '.DS_Store' in files: files.remove('.DS_Store')  ## not sure if this only happens in linux

        for ff, filename in enumerate(files):
            print(filename)
            print(100.0*(ff+1)/len(files))
            df = np.load(os.path.join(datadir, filename))

            number_windows = df['winsBF'].shape[0]
            window_size = df['winsBF'].shape[2]

            for k in range(number_windows):

                if ff == 0 and k == 0:
                    # write header
                    for ii in range(window_size):
                        if ii == 0:
                            header = ['Filename', 'Patient ID', 'Read', 'Window Index', 'ICP', 'Window Value ' + str(ii)]
                        else:
                            header.append('Window Value ' + str(ii))
                    writer.writerow(header)

                winsBF = df['winsBF'][k, 0, :].tolist()
                row = [filename, filename.split('_')[0], filename.split('_')[1], df['winsIndex'][k], df['ICPmean'][k]] + winsBF

                writer.writerow(row)

data_to_csv(source_datadir, source_filename)
data_to_csv(target_datadir, target_filename)