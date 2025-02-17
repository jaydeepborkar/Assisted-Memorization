import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def extract_emails(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))


def initialize_memorized_emails():
    return {
        'topk': {'e1': {}, 'e2': {}, 'e3': {}},
        'greedy': {'e1': {}, 'e2': {}, 'e3': {}}
    }


def calculate_memorized_emails():
    
    memorized_emails = initialize_memorized_emails()
    sampling_methods = ['topk', 'greedy']
    epochs = ['e1', 'e2', 'e3']

    for method in sampling_methods:
        for epoch in epochs:
            for i in range(10, 110, 10):
                emails = extract_emails(f"datasets/{i}.txt")

                memorized_emails[method][epoch][i] = []

                for j in range(10, 110, 10):
                    if j < i:
                        memorized_emails[method][epoch][i].append(np.nan)  
                        continue

                    gen_emails = extract_emails(f"generations/XL/{method}/seed42_A/txt/{epoch}/{j}.txt")
                    intersection = len(emails & gen_emails)
                    memorized_emails[method][epoch][i].append(intersection)

    return memorized_emails


def create_plot(memorized_emails, method):
    """
    Creates a heatmap for all models.
    """
    col_labels = [f"M{n}" for n in range(1, 11)]
    row_labels = [f"D{n*10}%" for n in range(1, 11)]
    epoch_titles = ['Epoch 1', 'Epoch 2', 'Epoch 3']
    epochs = ['e1', 'e2', 'e3']

    fig, axes = plt.subplots(1, 3, figsize=(32, 14), sharey=True, sharex=True)   
    
    for idx, epoch in enumerate(epochs):
        data = np.array(list(memorized_emails[method][epoch].values()), dtype=float)
        
        annotations = []
        for i, row in enumerate(data):
            annot_row = []
            for j, value in enumerate(row):
                if np.isnan(value):  
                    annot_row.append('-')
                else:   
                    annot_row.append(int(value))
            annotations.append(annot_row)

        ax = axes[idx]
        
        sns.heatmap(data, ax=ax, cmap='Blues', cbar=False, square=True, linewidths=0.5, 
                    linecolor='gray', xticklabels=col_labels, yticklabels=row_labels, 
                    vmin=0, vmax=np.nanmax(data), annot=annotations, fmt='s',  
                    annot_kws={"size": 20})
        
        ax.set_title(epoch_titles[idx], fontsize=36, pad=20)
        ax.set_xlabel("Model", fontsize=36, labelpad=15)
        if idx == 0:
            ax.set_ylabel("Dataset", fontsize=36, labelpad=15)

        ax.tick_params(axis='both', which='major', labelsize=34)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=np.nanmax(data)))
        sm.set_array([]) 
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=30)

    plt.subplots_adjust(right=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.suptitle(f'{method.capitalize()} Decoding', fontsize=40, y=1.01)
    plt.savefig(f'plots/add_{method}_test.pdf', bbox_inches='tight')
    plt.show()


def main():

    memorized_emails = calculate_memorized_emails()
    create_plot(memorized_emails, 'topk')
    create_plot(memorized_emails, 'greedy')


if __name__ == "__main__":
    main()
