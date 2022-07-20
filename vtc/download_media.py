import argparse
import html
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Union
from urllib.error import HTTPError

import pandas as pd
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

_AUDIO_VARIATIONS = ["DASH_audio.mp4", "audio", "DASH_audio", "audio.mp4"]
_DOWNLOAD_EXTENSIONS = [".png", ".mp4", ".mp3"]
_VIDEO_VARIATIONS = [
    "DASH_1080.mp4",
    "DASH_1080",
    "DASH_720.mp4",
    "DASH_720",
    "DASH_480.mp4",
    "DASH_480",
    "DASH_360.mp4",
    "DASH_360",
    "DASH_240.mp4",
    "DASH_240",
    "DASH_96.mp4",
    "DASH_96",
    "DASH_4_8_M",
    "DASH_2_4_M",
    "DASH_1_2_M",
    "DASH_600_K",
]


class VTCVideoDownloader(object):
    """Downloads image previews and videos for the VTC dataset.
    Args:
        dataset_path (Union[str, Path]): path to public csv, where each post should include:
            - reddit_id: base 10 version of the base 36 reddit id
            - subreddit: the subreddit the post is in
            - video_url: url to video mp4 file
            - post_url: url to post
            - image_preview_url: url to a high resolution image preview of the video
            - video_length: duration of video
            - comment_ids: curated ids of comments
        workers (int, optional): number of workers to use in parallel when downloading the data.
            Defaults to 10.
        max_resolution (int, optional): max resolution of the video file. Defaults to 1080.
        audio_variations (List[str], optional): possible endings for the audio url.
            Defaults to _AUDIO_VARIATIONS.
        video_variations (List[str], optional): possible endings for the video url.
            Defaults to _VIDEO_VARIATIONS.
        download_extensions (List[str], optional): supported extensions of the files to be saved.
            Defaults to _DOWNLOAD_EXTENSIONS.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        workers: int = 12,
        max_resolution: int = 1080,
        audio_variations: List[str] = _AUDIO_VARIATIONS,
        video_variations: List[str] = _VIDEO_VARIATIONS,
        download_extensions: List[str] = _DOWNLOAD_EXTENSIONS,
    ):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path, dtype=str)
        self.workers = workers
        self.max_resolution = max_resolution
        self.audio_variations = audio_variations
        self.video_variations = video_variations
        self.download_extensions = download_extensions

    def download_file(
        self, url: str, id: int, save_to_folder: Path, extension: str = ".png"
    ) -> bool:
        """Downloads file from url and saves it with the given extension.

        Args:
            url (str): post url
            id (int): reddit id to be used as filename
            save_to_folder (Path): which folder to save file in
            extension (str, optional): extension to save file with. Defaults to ".png".

        Returns:
            bool: whether the file was successfully downloaded
        """
        assert (
            extension in self.download_extensions
        ), f"Format {extension} not supported."
        download_success = False
        try:
            filename = save_to_folder / Path(f"{id}{extension}")
            urllib.request.urlretrieve(html.unescape(url), filename)
            download_success = True
        except HTTPError as e:
            logging.debug(f"failed to download file for url {url} with error {e}")
        return download_success

    def download_video(self, url: str, id: int, save_to_folder: Path) -> bool:
        """Downloads video file from url. If default url doesn't work, retries with
        other common resolutions found on Reddit.

        Args:
            url (str): post url
            id (int): reddit id to be used as filename
            save_to_folder (Path): which folder to save file to

        Returns:
            bool: whether the video was successfully downloaded
        """
        download_success = False
        download_success = self.download_file(url, id, save_to_folder, extension=".mp4")
        if not download_success:
            for size in self.video_variations:
                if isinstance(size, int) and size > self.max_resolution:
                    continue
                url_root = url.split("DASH_")[0]
                url_tmp = f"{url_root}{size}"
                download_success = self.download_file(
                    url_tmp, id, save_to_folder, extension=".mp4"
                )
                if download_success:
                    break
        if not download_success:
            logging.info(f"Couldn't download video {url}")
        return download_success

    def download_audio_file(self, url: str, id: int, save_to_folder: Path) -> bool:
        """Downloads audio file from url. If default url doesn't work, retries with
        other common endings for audio files found on Reddit.

        Args:
            url (str): post url
            id (int): reddit id to be used as filename
            save_to_folder (Path): which folder to save file to

        Returns:
            bool: whether the audio was successfully downloaded
        """
        url = url.split("DASH_")[0]
        try_urls = [f"{url}{s}" for s in self.audio_variations]
        for url in try_urls:
            download_success = self.download_file(
                url, id, save_to_folder, extension=".mp3"
            )
            if download_success:
                break
        if not download_success:
            logging.debug(f"Couldn't download audio {url}.")
        return download_success

    def download_files(
        self,
        urls: List[str],
        ids: List[int],
        save_to_folder: Path,
        download_func: callable,
        modality: str = "image",
    ):
        """Downloads either image, audio or video files in parallel to the directory specified in save_to_folder.

        Args:
            urls (List[str]): list of post urls
            ids (List[int]): list of post ids
            save_to_folder (Path): which folder to save file to
            modality (str, optional): type of file to save. Defaults to "image".
        """
        assert modality in [
            "image",
            "video",
            "audio",
        ], f"Modality {modality} not supported. Can only download images, videos or audio."

        download_status: List[bool] = []
        with tqdm(total=len(urls), desc=f"Downloading {modality}s") as pbar:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for _status in executor.map(
                    download_func, urls, ids, [save_to_folder] * len(urls)
                ):
                    download_status.append(_status)
                    pbar.update()
        logging.info(
            f"Done downloading {modality}s. Total downloaded {modality}s: {sum(download_status)}"
        )

    def download_previews(self, save_to_folder: Path):
        """Downloads all image previews of videos.
        Args:
            save_to_folder (Path): which folder to save image files to.
        """
        df = self.df[~self.df.image_preview_url.isna()]
        self.download_files(
            df.image_preview_url,
            df.reddit_id,
            save_to_folder,
            self.download_file,
            modality="image",
        )

    def download_videos(self, save_to_folder: Path):
        """Downloads all videos (without sound).
        Args:
            save_to_folder (Path): which folder to save video (and audio) files to.
        """
        df = self.df[~self.df.video_url.isna()]
        self.download_files(
            df.video_url,
            df.reddit_id,
            save_to_folder,
            self.download_video,
            modality="video",
        )

    def download_audio_files(self, save_to_folder: Path):
        """Downloads all audio files.
        Args:
            save_to_folder (Path): which folder to to save audio files to.
        """
        df = self.df[~self.df.video_url.isna()]
        self.download_files(
            df.video_url,
            df.reddit_id,
            save_to_folder,
            self.download_audio_file,
            modality="audio",
        )


def main(args):
    Path.mkdir(args.save_to_folder, exist_ok=True)
    downloader = VTCVideoDownloader(
        dataset_path=args.csv,
        workers=args.num_workers,
        max_resolution=args.max_resolution,
    )
    if args.download_preview:
        downloader.download_previews(args.save_to_folder)
    if args.download_video:
        downloader.download_videos(args.save_to_folder)
    if args.download_audio:
        downloader.download_audio_files(args.save_to_folder)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--csv",
        required=True,
        type=str,
        help="path to public VTC dataset csv (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="resume downloading",
    )
    args.add_argument(
        "-dp",
        "--download_preview",
        action="store_true",
        help="download image files",
    )
    args.add_argument(
        "-dv",
        "--download_video",
        action="store_true",
        help="download video files",
    )
    args.add_argument(
        "-da",
        "--download_audio",
        action="store_true",
        help="download audio files",
    )
    args.add_argument(
        "-nw",
        "--num_workers",
        default=12,
        type=int,
        help="number of workers to use when downloading files in parallel",
    )
    args.add_argument(
        "-res",
        "--max_resolution",
        default=1080,
        type=int,
        help="max resolution for videos",
    )
    args.add_argument(
        "-f",
        "--save_to_folder",
        default="vtc_media/",
        type=Path,
        help="folder where to download media",
    )
    args = args.parse_args()
    assert any(
        mod for mod in [args.download_audio, args.download_video, args.download_preview]
    ), "You must specify at least one modality to download."
    main(args)
