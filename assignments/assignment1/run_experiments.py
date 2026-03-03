"""Run all experiments: vary n_mels then groups, log results to CSV files."""
import os
import subprocess

PYTHON = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
HERE   = os.path.dirname(os.path.abspath(__file__))
LOG    = os.path.join(HERE, "training_log.csv")
ELOG   = os.path.join(HERE, "epoch_log.csv")


def run(n_mels, groups, epochs=10):
    print(f"\n=== n_mels={n_mels}  groups={groups} ===", flush=True)
    subprocess.run(
        [PYTHON, "-u", "train.py",
         "--n_mels", str(n_mels),
         "--groups", str(groups),
         "--epochs", str(epochs)],
        cwd=HERE, check=False
    )


def main():
    with open(LOG,  "w") as f:
        f.write("n_mels,groups,params,flops,test_acc,train_loss,epoch_time\n")
    with open(ELOG, "w") as f:
        f.write("n_mels,groups,epoch,train_loss,val_acc,epoch_time\n")

    # Phase 3: vary n_mels, groups=1
    for n_mels in [20, 40, 80]:
        run(n_mels, groups=1)

    # Phase 4: vary groups, n_mels=40 baseline
    for groups in [2, 4, 8, 16]:
        run(n_mels=40, groups=groups)


if __name__ == "__main__":
    main()
