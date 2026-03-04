import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_mels_experiments(df):
    mels_df = df[df['groups'] == 1].groupby('n_mels').mean().reset_index()
    
    # 1. n_mels vs testing accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(mels_df['n_mels'], mels_df['test_acc'], marker='o', linestyle='-', color='b')
    plt.title('Testing Accuracy vs Number of Mel Filterbanks (n_mels)')
    plt.xlabel('n_mels')
    plt.ylabel('Testing Accuracy')
    plt.grid(True)
    plt.xticks([20, 40, 80])
    plt.savefig('n_mels_vs_acc.png')
    plt.close()

def plot_groups_experiments(df):
    groups_df = df[df['n_mels'] == 40].groupby('groups').mean().reset_index()
    
    # 2. Epoch Training Time vs Groups
    plt.figure(figsize=(8, 5))
    plt.plot(groups_df['groups'], groups_df['epoch_time'], marker='s', linestyle='-', color='r')
    plt.title('Epoch Training Time vs Groups (n_mels=40)')
    plt.xlabel('Groups')
    plt.ylabel('Epoch Training Time (seconds)')
    plt.grid(True)
    plt.xticks([1, 2, 4, 8, 16])
    plt.savefig('groups_vs_time.png')
    plt.close()
    
    # 3. Number of parameters & FLOPs vs Groups
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Groups')
    ax1.set_ylabel('Number of Parameters', color=color)
    ax1.plot(groups_df['groups'], groups_df['params'], marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([1, 2, 4, 8, 16])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('FLOPs', color=color)
    ax2.plot(groups_df['groups'], groups_df['flops'], marker='x', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Parameters and FLOPs vs Groups')
    plt.grid(True)
    plt.savefig('groups_vs_params_flops.png')
    plt.close()

def main():
    if not os.path.exists('training_log.csv'):
        print("Error: training_log.csv not found!")
        return
        
    df = pd.read_csv('training_log.csv')
    
    # Clean up any potential formatting issues or empty lines
    df.dropna(inplace=True)
    
    print("Generating n_mels vs testing accuracy plot...")
    plot_mels_experiments(df)
    
    print("Generating group experiments plots...")
    plot_groups_experiments(df)
    
    print("Visualizations saved successfully!")

if __name__ == '__main__':
    main()
