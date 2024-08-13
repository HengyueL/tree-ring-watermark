import pickle
import os


if __name__ == "__main__":
    data_dir = os.path.join(
        "Result-Interm", "Tree-Ring",
        "Gustavosta", "vae",
        "cheng2020-anchor", "Img-1.pkl"
    )
    with open(data_dir, 'rb') as handle:
        data_dict = pickle.load(handle)
    print()
    #  dict_keys(['index', 'interm_recon', 'watermark_gt_str'])