"""Global reproducibility."""
import os, random
import numpy as np

def seed_everything(seed: int = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
