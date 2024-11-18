import os
import shutil
import pandas as pd
import numpy as np
import subprocess

def load_datasets_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    url_prefix = 'https://dataset-ppopp.oss-cn-beijing.aliyuncs.com/dataset/'
    for index, row in data.iterrows():
        dataset_name = row['dataSet'] + '.npz'
        url = url_prefix + dataset_name
        
        #Download the dataset
        command = ["wget", "-O", dataset_name, url]
        subprocess.run(command)
        print(f"Downloaded {dataset_name} from {url}.")
        
    # #Download IGB-large
    # url_prefix = 'https://dataset-ppopp.oss-cn-beijing.aliyuncs.com/dataset/'
    # dataset = [ 'IGB_large']
    # for data in dataset:
    #     dataset_name = data + '.npz'
    #     url = url_prefix + dataset_name
    #     command = ["wget", "-O", dataset_name, url]
    #     subprocess.run(command)
    #     print(f"Downloaded {dataset_name} from {url}.")
        
    #Download the accuracy dataset
    url_prefix = 'https://dataset-ppopp.oss-cn-beijing.aliyuncs.com/dataset1/'
    dataset = [ 'cora', 'ell-acc', 'pubmed', 'question', 'min']
    for data in dataset:
        dataset_name = data + '.npz'
        url = url_prefix + dataset_name
        command = ["wget", "-O", dataset_name, url]
        subprocess.run(command)
        print(f"Downloaded {dataset_name} from {url}.")
        
# Main
csv_path = "data_filter.csv"  # CSV 文件路径
load_datasets_from_csv(csv_path)
