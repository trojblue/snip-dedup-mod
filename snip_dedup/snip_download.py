"""snip download"""

import requests
import os
import os.path
import fire
import numpy as np
import pandas as pd


def snip_download(outfolder="data/downloaded", start=0, end=2313, dl_dedup_set=True):
    """Download and deduplicate a dataset.

    Parameters
    ----------
    outfolder : str, optional
        Where to put the downloaded metadata
    start : int, optional
        Start index of the metadata
    end : int, optional
        End index of the metadata
    dl_dedup_set : bool, optional
        Indicate whether you'll download the dedup set again (2GB)
    """
    metadata_dir = os.path.join(outfolder, "metadata")
    dedup_set_path = os.path.join(
        outfolder, "is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy"
    )
    os.makedirs(metadata_dir, exist_ok=True)

    if dl_dedup_set:
        print("downloading dedup set...")
        url = "https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy"
        response = requests.get(url)
        open(dedup_set_path, "wb").write(response.content)

    is_dup_all = np.load(dedup_set_path).ravel()
    abs_ind = 0

    total_parquet_len = 0
    total_dupe_len = 0
    for n in range(start, end):
        print(f"downloading metadata file {n}/{end}")
        url = f"https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/metadata/metadata_{n:04d}.parquet"
        response = requests.get(url)
        parquet_path = os.path.join(metadata_dir, f"metadata_{n:04d}.parquet")
        open(parquet_path, "wb").write(response.content)

        # perform the deduplication
        md = pd.read_parquet(parquet_path)
        dup_chunk = is_dup_all[abs_ind: abs_ind + len(md.index)]

        # 读取的parquet文件的长度
        parquet_len = len(md.index)

        # 读取的parquet文件的长度对应的is_dup_all
        curr_dup_chunk = is_dup_all[abs_ind: abs_ind + parquet_len]

        # take only duped
        md = md[dup_chunk]

        # todo: DEBUG
        curr_dup_total = np.sum(curr_dup_chunk)
        md_len = len(md.index)
        total_parquet_len += parquet_len
        total_dupe_len += md_len

        # overwrite metadata
        md.to_parquet(parquet_path)
        abs_ind += parquet_len

    print(f"total_parquet_len: {total_parquet_len}")
    print(f"total_dupe_len: {total_dupe_len}")
    print("DONE")


def debug():
    """Debug the snip_dedup package."""
    snip_download(outfolder="data/downloaded", start=0, end=20, dl_dedup_set=False)


if __name__ == "__main__":
    fire.Fire(snip_download)
    # debug()
