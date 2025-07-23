import subprocess

BACKBONE_ARCHITECTURE = [
    "deit_tiny_patch16_224", 
    "deit_small_patch16_224", 
    "cait_xxs24_224"
]

PROTOTYPE_DISTRIBUTION = [
    # "in_distribution",         # train
    "out_of_distribution_birds", # finetune, attack
    "cars"                       # finetune, attack
]

for backbone_architecture in BACKBONE_ARCHITECTURE:
    for prototype_distribution in PROTOTYPE_DISTRIBUTION:
        for random_seed in range(0, 5):
            subprocess.run([
                "sbatch", 
                "unsafe_sbatch.sh", 
                backbone_architecture, 
                prototype_distribution, # finetune, attack
                str(random_seed)
            ])