import subprocess
import os
from itertools import product

def run_experiment(n_mels, groups):
    print(f"=== Running experiment: n_mels={n_mels}, groups={groups} ===")
    cmd = [
        "python3", "train.py", 
        "--n_mels", str(n_mels), 
        "--groups", str(groups), 
        "--epochs", "2"  # short epochs for report speed
    ]
    subprocess.run(cmd)

def main():
    if os.path.exists("training_log.csv"):
        os.remove("training_log.csv")
        
    with open("training_log.csv", "w") as f:
        f.write("n_mels,groups,params,flops,test_acc,train_loss,epoch_time\n")
        
    print("--- Stage 1: Varying n_mels ---")
    for n_mels in [20, 40, 80]:
        run_experiment(n_mels=n_mels, groups=1)
        
    print("--- Stage 2: Varying groups ---")
    baseline_n_mels = 40 # arbitrary baseline
    for groups in [2, 4, 8, 16]:
        run_experiment(n_mels=baseline_n_mels, groups=groups)

if __name__ == '__main__':
    main()
