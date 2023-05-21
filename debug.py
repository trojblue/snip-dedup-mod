import os
import os.path
import fire
import pandas as pd
from snip_dedup.snip_download import snip_download
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def get_dup_array():
    outfolder = "data/downloaded"
    metadata_dir = os.path.join(outfolder, "metadata")
    dedup_set_path = os.path.join(
        outfolder, "is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy"
    )
    os.makedirs(metadata_dir, exist_ok=True)

    is_dup_all = np.load(dedup_set_path).ravel()
    return is_dup_all


def plot_dup_array(chunk_size=1000000):
    is_dup_all = get_dup_array()

    # determine the number of chunks
    num_chunks = len(is_dup_all) // chunk_size

    # create empty lists to hold the sums and percentages of True values for each chunk
    sums = []
    percentages = []

    total_true = 0
    for i in range(num_chunks):
        chunk = is_dup_all[i * chunk_size: (i + 1) * chunk_size]
        true_count = np.sum(chunk)
        percentage = (true_count / chunk_size) * 100  # calculate percentage
        sums.append(true_count)
        percentages.append(percentage)
        total_true += true_count

    np_true = np.sum(is_dup_all)

    # create an array of chunk numbers for the x-axis (1-indexed)
    chunks = np.arange(1, num_chunks + 1)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Number of duplicates per {chunk_size}-item chunk",
                                                        f"Percentage of duplicates per {chunk_size}-item chunk"))

    # Number of duplicates
    fig.add_trace(go.Scatter(x=chunks, y=sums, mode='lines+markers', name='Number of duplicates'), row=1, col=1)
    fig.update_xaxes(title_text="Chunk number", row=1, col=1)
    fig.update_yaxes(title_text="Number of duplicates", row=1, col=1)

    # Percentage of duplicates
    fig.add_trace(go.Scatter(x=chunks, y=percentages, mode='lines+markers', name='Percentage of duplicates'), row=1,
                  col=2)
    fig.update_xaxes(title_text="Chunk number", row=1, col=2)
    fig.update_yaxes(title_text="Percentage", row=1, col=2, range=[0, 100])  # adjust this range if needed

    fig.show()


def debug():
    """Debug the snip_dedup package."""
    snip_download(outfolder="data/downloaded", start=0, end=20, dl_dedup_set=False)


if __name__ == '__main__':
    plot_dup_array()
