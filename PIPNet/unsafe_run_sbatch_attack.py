import subprocess

NET = [
    "resnet18",
    "convnext_tiny_13",
    "resnet50",
]

MODES = [
    "disguising",
    "redherring"
]

for net in NET:
    for mode in MODES:
        for random_seed in range(0, 3):
            subprocess.run([
                "sbatch", 
                "unsafe_sbatch_attack.sh", 
                net, 
                str(random_seed),
                mode
            ])