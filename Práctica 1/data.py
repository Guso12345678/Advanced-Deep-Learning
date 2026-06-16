import os
import sys
import zipfile
import torch
import pandas as pd
from torch.utils.data import TensorDataset


class WineQuality(TensorDataset):
    def __init__(self, root, normalize=True, train=None):
        self.root = os.path.abspath(root)
        self._folder = os.path.join(self.root, "winequality")

        self._ensure_data_available()

        df = pd.concat(
            [
                pd.read_csv(
                    os.path.join(self._folder, "winequality-red.csv"), delimiter=";"
                ),
                pd.read_csv(
                    os.path.join(self._folder, "winequality-white.csv"), delimiter=";"
                ),
            ]
        )
        split_file = os.path.join(self._folder, "winequality_split.pt")

        x = torch.tensor(df.loc[:, df.columns != "quality"].values, dtype=torch.float32)
        quality = torch.tensor(df["quality"].values, dtype=torch.float32)

        if train is not None:
            if os.path.exists(split_file):
                rp = torch.load(split_file)
            else:
                torch.manual_seed(1)
                rp = torch.randperm(x.size(0))
                torch.save(rp, split_file)

            n_train = int(x.size(0) * 0.8)
            if train:
                x = x[rp[:n_train]]
                quality = quality[rp[:n_train]]
            else:
                x = x[rp[n_train:]]
                quality = quality[rp[n_train:]]

        if normalize:
            self.x_mean = x.mean(dim=0)
            self.x_std = x.std(dim=0)
            x = (x - self.x_mean) / self.x_std

        super(WineQuality, self).__init__(x, quality)

    def _ensure_data_available(self):
        if self._check_integrity():
            return

        in_colab = "google.colab" in sys.modules
        colab_zip_path = "/content/wine+quality.zip"

        if in_colab and os.path.exists(colab_zip_path):
            print(f"Found zip file at {colab_zip_path}. Extracting...")
            os.makedirs(self._folder, exist_ok=True)
            with zipfile.ZipFile(colab_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.root)
            print(f"Extracted to: {self.root}")
            if not self._check_integrity():
                raise FileNotFoundError(
                    f"Extraction complete, but dataset files not found under '{self._folder}'. "
                    "Please verify the zip contents."
                )
            return

        raise FileNotFoundError(f"WineQuality dataset not found in '{self._folder}'.")

    def _check_integrity(self):
        required = ["winequality-red.csv", "winequality-white.csv"]
        return all(os.path.isfile(os.path.join(self._folder, f)) for f in required)
