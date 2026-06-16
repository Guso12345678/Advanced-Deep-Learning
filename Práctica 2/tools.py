import os
import sys
import torch


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


def save_model(model, filename, confirm=True):
    if confirm:
        try:
            save = input(
                "Do you want to save the model (type yes to confirm)? "
            ).lower()
            if save != "yes":
                print("Model not saved.")
                return
        except:
            raise Exception(
                "The notebook should be run or validated with skip_training=True."
            )

    torch.save(model.state_dict(), filename)
    print("Model saved to %s." % (filename))


def load_model(model, filename, device):
    model.load_state_dict(
        torch.load(filename, map_location=lambda storage, loc: storage)
    )
    print("Model loaded from %s." % filename)
    model.to(device)
    model.eval()
