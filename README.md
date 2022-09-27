# VTC: a curated Reddit dataset of Videos, Titles, and Comments

Welcome to the VTC dataset repository! Here you will find code & instructions on how to download the dataset from our paper "VTC: Improving Video-Text Retrieval with User Comments". Each instance represents a public video post with caption and comments from reddit.com.
# Intended use case

This dataset was created for research purposes. More specifically,this dataset addresses the research problem of using a weakly informative modality (user comments) in conjunction with other learning signals such as titles and videos for learning multi-modal representations.

# Dataset curation

This dataset is a sample of a larger, unfiltered version of the original dataset that we have collected. From the initial
version, we handpicked a list of ”safe” subreddits and removed posts if: 1) they had the ”NSFW” or ”over 18” tags; 2) the videos contained faces or the captions contained toxic or offensive text.

We are only publicly releasing urls such that if a user decides to remove a post, the link to the post will become invalid. This dataset should not be used for tasks that might disclose the identity of the users or directly or indirectly harm them.

More details about the dataset can be found in `DATASHEET.md`, where we answer the questions proposed by [Gebru et al.](https://arxiv.org/abs/1803.09010), which were introduced as a way of documenting new datasets.

# Installation

In order to install the conda environment, [Anaconda](https://conda.io/docs/user-guide/install/download.html) will need to be installed first.


```bash
# Clone the repository
git clone https://github.com/unitaryai/VTC
cd VTC

# Create a new conda environment
conda create -n vtc python>=3.10
conda activate vtc

# Install dependencies
pip install -r requirements.txt
```

# Usage

### Download the public csv file containing:
- `reddit_id`: base 10 version of the base 36 reddit id
- `post_url`: url of the reddit post
- `subreddit`: the subreddit the post is in
- `video_url`: url to video mp4 file
- `image_preview_url`: url to a high resolution image preview of the video
- `video_length`: duration of video in seconds
- `comment_ids`: curated ids of comments and hierarchical depth levels within a comment thread

```bash
wget https://github.com/unitaryai/VTC/releases/download/v0.1.0-alpha/VTC_v1.0_public.csv.tar.gz
tar -xvzf VTC_v1.0_public.csv.tar.gz
```

### Download the metadata

```bash
# download metadata including titles and comments
python vtc/download_metadata.py --csv $PUBLIC_CSV --save_to $SAVE_TO_CSV
```
If the download breaks unexpectedly, resume download by adding `--resume`.

### Download the media files

```bash
# download image previews
python vtc/download_media.py --csv $PUBLIC_CSV --save_to_folder $FOLDER --download_preview

# download videos
python vtc/download_media.py --csv $PUBLIC_CSV --save_to_folder $FOLDER --download_video

# download audio files
python vtc/download_media.py --csv $PUBLIC_CSV --save_to_folder $FOLDER --download_audio
```

# Citation


```text
@inproceedings{hanu2022vtc,
    title={VTC: Improving Video-Text Retrieval with User Comments},
    author={Laura Hanu and James Thewlis and Yuki M. Asano and Christian Rupprecht},
    booktitle={ECCV},
    year={2022}
}
```
