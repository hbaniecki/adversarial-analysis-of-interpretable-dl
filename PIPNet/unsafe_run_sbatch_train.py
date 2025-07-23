import subprocess

NET = [
    "resnet18",
    "convnext_tiny_13", 
    "resnet50",
]

for net in NET:
    for random_seed in range(0, 3):
        subprocess.run([
            "sbatch", 
            "unsafe_sbatch_train.sh", 
            net, 
            str(random_seed)
        ])