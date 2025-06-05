from sklearn.datasets import make_blobs
import time
import pandas as pd
import os


def get_dataset(n_samples=2000, n_features=2):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=2, random_state=0)
    df = pd.DataFrame(X)
    df.columns = ["f1", "f2"]
    df["label"] = y
    return df


def save_dateset(dataset, prefix="datasets/", filename="blobs.csv"):
    os.makedirs(os.path.dirname(f"{prefix}/{filename}"), exist_ok=True)
    dataset.to_csv(f"{prefix}/{filename}", index=False)
    return None


if __name__ == '__main__':
    print('Generating dataset...')
    df = get_dataset()
    print('Saving dataset...')
    save_dateset(df)
    print('Done!')


