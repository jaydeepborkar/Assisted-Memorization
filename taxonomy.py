"""
Generate a taxonomy of memorization during continuous training 
"""


import re 
import argparse
import os 
import pandas as pd 
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
from matplotlib.ticker import FuncFormatter, LogLocator


def process_emails(epoch_files):
    """
    get new emails seen between each checkpoint, emails seen in the past, total # of emails seen until each checkpoint, and the actual email set. 
    
    """
    newemails_ckpt = {}
    total_count = 0
    total_emails = {}
    emails_dict = {}
    prev_emails = {}

    for i in range(10, 101, 10):
        with open(epoch_files[i], "r") as file:
            text = file.read()
            emails_new = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            newemails_ckpt[i] = set(emails_new)  
            total_count += len(newemails_ckpt[i])

    prev_key = None
    for key in sorted(newemails_ckpt.keys()):
        if prev_key is not None:
            emails_buffer = []
            count = 0
            for k in range(10, key + 10, 10):
                emails_buffer.extend(newemails_ckpt[k])
                count += len(newemails_ckpt[k])
            total_emails[key] = count
            emails_dict[key] = emails_buffer
        else:
            total_emails[key] = len(newemails_ckpt[key])
            emails_dict[key] = newemails_ckpt[key]
        prev_key = key

    prev_emails[10] = list(newemails_ckpt[10])
    for key in range(20, 101, 10):
        union_list = []
        for inner_key in range(10, key + 1, 10):
            union_list.extend(list(newemails_ckpt[inner_key]))
        prev_emails[key] = union_list

    return total_emails, emails_dict, prev_emails, newemails_ckpt


def extract_emails_from_file(file_path):
    """simple function to extract emails from a txt using regex"""
    with open(file_path, "r") as file:
        text = file.read()
    return set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))


def process_email_files(base_path, indices, reference_emails=None, emails_dict=None):
    """compute memorization by comparing generations with actual data"""
    memorized_emails = {}
    
    for i in indices:
        emails = extract_emails_from_file(f"{base_path}/{i}.txt")

        key_numeric = int(re.match(r"(\d+)", str(i)).group(1)) 

        if emails_dict:
            memorized_emails[key_numeric] = emails & set(emails_dict[key_numeric]) #for first epoch, we compare generation with the data seen until that checkpoint. 
        elif reference_emails:
            memorized_emails[key_numeric] = emails & reference_emails
    
    return memorized_emails


def compute_immediate_memorized(memorized_emails, newemails_ckpt, epoch_num):
    """computes immediate memorization at the end of 10% training"""
    if epoch_num == 1:
        return {10: set(memorized_emails[10])}
    
    return {10: set(memorized_emails[10]) & set(newemails_ckpt[10])}


def compute_retained_memorized(memorized_emails, prev_epoch_emails, epoch_num):
    """computes retained memorization at the end of 10% training"""
    if epoch_num == 1:
        return {10: set()}
    
    return {10: set(memorized_emails[10]) & set(prev_epoch_emails[100])}


def compute_assisted_memorized(memorized_emails, immediate_elements, retained_elements):
    """computes assisted memorization at the end of 10% training"""
    assisted_set = set(memorized_emails[10])
    assisted_set.difference_update(immediate_elements)
    assisted_set.difference_update(retained_elements)
    return {10: assisted_set}


def compute_forgotten(prev_epoch_emails, retained_elements):
    """computes forgotten memorization at the end of 10% training"""
    forgotten_set = set(prev_epoch_emails[100] if prev_epoch_emails else set())
    forgotten_set.difference_update(retained_elements)
    return {10: forgotten_set}


def compute_more_memorizarion(memorized_emails, prev_emails, newemails_ckpt, epoch_num):
    """computes the above categories of memorization for the remainder of the training (after 10%)"""
    retained_memorized = {}
    assisted_memorized = {}
    immediate_memorized = {}
    forgotten = {}
    
    keys = sorted(memorized_emails.keys())
    prev_key = None
    
    for key in keys:
        if key != 10:
            retained_elements = set(memorized_emails[key]) & set(memorized_emails[prev_key]) #emails stay memorized across both checkpoints in retained memorization 
            retained_memorized[key] = retained_elements
            
            immediate_set = set(memorized_emails[key]) & set(newemails_ckpt[key]) #compare generations with the "new" data seen to get immediate memorization 
            immediate_memorized[key] = immediate_set
            
            assisted_set = set(memorized_emails[key]) #the rest of memorization is assisted 
            assisted_set.difference_update(retained_elements)
            assisted_set.difference_update(immediate_set)
            
            if epoch_num == 1 and prev_emails:
                assisted_set = assisted_set & set(prev_emails[prev_key]) #sanity check makes sure assisted memorized emails were previously seen but not memorized 
            
            assisted_memorized[key] = assisted_set
            
            forgotten_set = set(memorized_emails[prev_key]) #emails that get forgotten at next checkpoint 
            forgotten_set.difference_update(retained_elements)
            forgotten[key] = forgotten_set
        
        prev_key = key
    
    return retained_memorized, assisted_memorized, immediate_memorized, forgotten


def compute_memorization(memorized_emails, newemails_ckpt, epoch_num, prev_epoch_emails=None):
    """
    main function to compute various categories of memorization.
    """ 
   
    immediate_memorized = compute_immediate_memorized(memorized_emails, newemails_ckpt, epoch_num)
    retained_memorized = compute_retained_memorized(memorized_emails, prev_epoch_emails, epoch_num)
    assisted_memorized = compute_assisted_memorized(memorized_emails, immediate_memorized[10], retained_memorized[10])
    forgotten = compute_forgotten(prev_epoch_emails, retained_memorized[10])
    
    additional_retained, additional_assisted, additional_immediate, additional_forgotten = compute_more_memorizarion(
        memorized_emails, prev_epoch_emails, newemails_ckpt, epoch_num
    )
    
    retained_memorized.update(additional_retained)
    assisted_memorized.update(additional_assisted)
    immediate_memorized.update(additional_immediate)
    forgotten.update(additional_forgotten)
    
    return retained_memorized, assisted_memorized, immediate_memorized, forgotten


def create_dataframe(keys, immediate_memorized, assisted_memorized, retained_memorized, forgotten, total_emails):
    """
    create a df for plotting purpose. 
    """ 
    data = [
        {
            'split': f"{key}%",
            'immediate_memorized': immediate_memorized.get(key, set()),
            'assisted_memorized': assisted_memorized.get(key, set()),
            'retained_memorized': retained_memorized.get(key, set()),
            'forgotten': forgotten.get(key, set()),
            #'total_memorization': total_memorization.get(key, set()), 
            #'generated_emails': generated_emails.get(key, set()), 
            'total_emails': total_emails.get(key, set()), 
        }
        for key in keys
    ]
    return pd.DataFrame(data)


def plot_memorization_progress(dfs, fig_size=(26, 7)):
    fig, axes = plt.subplots(1, 3, figsize=fig_size, sharey=True)
    axes[0].set_ylabel('% Extracted', fontsize=26, labelpad=20)

    handles_all = []
    labels_all = []

    min_percent_forgotten = float('inf')
    max_percent_forgotten = float('-inf')

    def format_func(value, tick_number):
        return f'{value:.2f}'.rstrip('0').rstrip('.')

    def calculate_percentages(df, reference_df=None):
        df['immediate_memorized_count'] = df['immediate_memorized'].apply(len)
        df['assisted_memorized_count'] = df['assisted_memorized'].apply(len)
        df['retained_memorized_count'] = df['retained_memorized'].apply(len)
        df['forgotten_count'] = df['forgotten'].apply(len)

        if reference_df is None:
            total_emails = df['total_emails']
        else:
            total_emails = reference_df.iloc[9]['total_emails']

        df['immediate_memorized_percent'] = (df['immediate_memorized_count'] / total_emails) * 100
        df['assisted_memorized_percent'] = (df['assisted_memorized_count'] / total_emails) * 100
        df['retained_memorized_percent'] = (df['retained_memorized_count'] / total_emails) * 100
        df['forgotten_percent'] = (df['forgotten_count'] / total_emails) * 100

    #calculate memorization percentages for each dataframe  
    for i, df in enumerate(dfs):
        reference_df = dfs[0] if i > 0 else None  #for e1, normalize by the emails seen until each checkpoint. For e2 and e3, normalize by total emails seen in e1. 
        calculate_percentages(df, reference_df)

    for i, df in enumerate(dfs, start=1):
        splits = df['split']
        assisted_memorized = df['assisted_memorized_percent']
        immediate_memorized = df['immediate_memorized_percent']
        retained_memorized = df['retained_memorized_percent']
        forgotten = df['forgotten_percent']
        data = [assisted_memorized, immediate_memorized, retained_memorized, forgotten]
        categories = ['Assisted', 'Immediate', 'Retained', 'Forgotten']

        stack = axes[i-1].stackplot(splits, *data, labels=categories)

        axes[i-1].set_title(f'Epoch {i}', fontsize=24, pad=10)

        axes[i-1].set_xlabel('% of Epoch', fontsize=26, labelpad=20)
        axes[i-1].grid(True)
        axes[i-1].tick_params(axis='both', labelsize=22)
        axes[i-1].yaxis.set_major_formatter(FuncFormatter(format_func))

        if i == 1:
            for element in stack:
                handles_all.append(element)
            labels_all.extend(categories)

        x_ticks = [f'{j}%' for j in range(10 + (i - 1) * 100, 101 + (i - 1) * 100, 10)]
        axes[i-1].set_xticks(range(len(x_ticks)))
        axes[i-1].set_xticklabels(x_ticks, rotation=45, ha='right', fontsize=22)

    fig.legend(handles_all, labels_all, loc='upper center', fontsize=24, ncol=5, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 0.80])  #(left, bottom, right, top)
    plt.savefig('plots/taxonomy.pdf', bbox_inches='tight')
    plt.show()


def main():
    
    epoch_files_e1 = {i: f"data/XL/seed42_B/epoch1_{i}%.txt" for i in range(10, 101, 10)}
    epoch_files_e2 = {i: f"data/XL/seed42_B/epoch2_{i}%.txt" for i in range(10, 101, 10)}
    epoch_files_e3 = {i: f"data/XL/seed42_B/epoch3_{i}%.txt" for i in range(10, 101, 10)}

    total_emails_e1, emails_dict_e1, prev_emails_e1, newemails_ckpt_e1 = process_emails(epoch_files_e1)
    total_emails_e2, emails_dict_e2, prev_emails_e2, newemails_ckpt_e2 = process_emails(epoch_files_e2)
    total_emails_e3, emails_dict_e3, prev_emails_e3, newemails_ckpt_e3 = process_emails(epoch_files_e3)
    
    base_dataset_path = "datasets/100.txt"
    base_generation_path = "generations/XL/greedy/seed42_B/txt"
    indices = range(10, 110, 10)
    
    emails1 = extract_emails_from_file(base_dataset_path)
    
    memorized_emails_e1 = process_email_files(base_generation_path, [f"{i}e1" for i in indices], emails_dict=emails_dict_e1)                                                         
    memorized_emails_e2 = process_email_files(base_generation_path, [f"{i}e2" for i in indices], reference_emails=emails1) 
    memorized_emails_e3 = process_email_files(base_generation_path, [f"{i}e3" for i in indices], reference_emails=emails1) 
    
    retained_memorized_e1, assisted_memorized_e1, immediate_memorized_e1, forgotten_e1 = compute_memorization(
    memorized_emails_e1, newemails_ckpt_e1, epoch_num=1)

    retained_memorized_e2, assisted_memorized_e2, immediate_memorized_e2, forgotten_e2 = compute_memorization(
    memorized_emails_e2, newemails_ckpt_e2, epoch_num=2, prev_epoch_emails=memorized_emails_e1)

    retained_memorized_e3, assisted_memorized_e3, immediate_memorized_e3, forgotten_e3 = compute_memorization(
    memorized_emails_e3, newemails_ckpt_e3, epoch_num=3, prev_epoch_emails=memorized_emails_e2)
    
    keys = sorted(memorized_emails_e1.keys())

    df_e1 = create_dataframe(keys, immediate_memorized_e1, assisted_memorized_e1, retained_memorized_e1, forgotten_e1, total_emails_e1)
    df_e2 = create_dataframe(keys, immediate_memorized_e2, assisted_memorized_e2, retained_memorized_e2, forgotten_e2, total_emails_e2)
    df_e3 = create_dataframe(keys, immediate_memorized_e3, assisted_memorized_e3, retained_memorized_e3, forgotten_e3, total_emails_e3)

    df_e1.to_csv('dataframes/taxonomy_e1.csv', index=False) #saves dataframe of memorized examples. 
    df_e2.to_csv('dataframes/taxonomy_e2.csv', index=False)
    df_e3.to_csv('dataframes/taxonomy_e3.csv', index=False)
    
    plot_memorization_progress([df_e1, df_e2, df_e3])
                                                                                                                             
if __name__ == '__main__':
    main()
    
