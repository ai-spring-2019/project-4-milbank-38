
import sys
from sklearn import preprocessing
import numpy as np
import pandas as pd

def normalize(df, targets):
    res = df.copy()
    for name in df.columns:
        if name not in targets:
            max_value = df[name].max()
            min_value = df[name].min()
            res[name] = (df[name] - min_value) / (max_value - min_value)
        else:
            res[name] = df[name]
    return res
    
def main():
    file = sys.argv[1]

    df = pd.read_csv(file, sep=",")

    num_targets = 0
    for i in df.columns:
        if i.startswith("target"):
            num_targets += 1

    target_names = df.columns[-num_targets]

    norm_data = normalize(df, target_names)

    norm_data.to_csv("normalized_" + file, index=False)


if __name__ == '__main__':
    main()