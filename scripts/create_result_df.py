import pandas as pd
import re
import pickle
# import pickle5 as pickle
import argparse
import os
from tabulate import tabulate

def acc_per_cat(data):
    df_list = []
    for key in data.keys():
        df_list.append((bool(data[key]['correct']), data[key]['category']))
    df = pd.DataFrame(df_list, columns=['correct', 'category'])
    df['n_items'] = df['correct']
    df = df.groupby('category').agg({'correct': 'sum', 'n_items': 'count'})
    df['accuracy'] = (df['correct'] / df['n_items']) * 100
    df = df.drop(columns=['correct', 'n_items'])
    df = df.transpose().reset_index(drop=True)
    df.columns.name = None
    return df

def create_results_df(resfolder, dataset):
    '''
    resdir: location where all the .pckl files are stored
    dataset: choose one of 'SCAN', 'BATS' or 'WEBB' in uppercase.
    '''
    resdir = os.path.join(os.path.dirname(os.path.abspath('create_result_df.py')), '..', resfolder)
    df_rows = []
    for filename in os.listdir(resdir):
        if filename.endswith('.pckl'):
            if dataset in filename:
                file_path = os.path.join(resdir, filename)
                model = re.search(r'^(.*?)_', filename).group(1)
                method = re.search(r'_(.*?)_', filename).group(1)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                acc_row = acc_per_cat(data)
                info_row = pd.DataFrame(data={'model':[model], 'method':[method]})
                row = pd.concat([info_row, acc_row], axis=1)
                df_rows.append(row)
    df = pd.concat(df_rows, axis=0)
    return df

argParser = argparse.ArgumentParser()

argParser.add_argument("--resfolder") 
argParser.add_argument("--dataset", choices=['SCAN', 'BATS', 'WEBB']) 
args = argParser.parse_args() 

# used python .\create_result_df.py --resfolder 'results_new' --dataset 'WEBB'
print(tabulate(create_results_df(args.resfolder, args.dataset), headers='keys', tablefmt='.5f'))
         