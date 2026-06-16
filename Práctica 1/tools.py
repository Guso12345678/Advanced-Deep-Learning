import os
import sys


def select_data_dir(default_local="data"):
    in_colab = "google.colab" in sys.modules

    if os.path.isdir("/coursedata"):
        data_dir = "/coursedata"
    elif in_colab and os.path.isdir("/content/drive/MyDrive"):
        data_dir = "/content/drive/MyDrive"
    elif in_colab and os.path.isdir("/content"):
        data_dir = "/content"
    else:
        data_dir = default_local

    data_dir = os.path.abspath(data_dir)
    print(f"The data directory is: {data_dir}")
    return data_dir
