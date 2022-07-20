import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
from requests.exceptions import HTTPError

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_random
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

_METADATA_FIELDS = [
    "reddit_id",
    "subreddit",
    "video_url",
    "title",
    "post_url",
    "image_preview_url",
    "video_length",
    "comments",
]


class VTCMetadataDownloader(object):
    """Downloads metadata for the VTC dataset to a csv file.
    Args:
        dataset_path (Union[str, Path]): path to public csv, where each post should include:
            - reddit_id: base 10 version of the base 36 reddit id
            - subreddit: the subreddit the post is in
            - video_url: url to video mp4 file
            - post_url: url to post
            - image_preview_url: url to a high resolution image preview of the video
            - video_length: duration of video
            - comment_ids: curated ids of comments
        save_to (Union[str, Path]): path to csv file to save the extra metadata.
        metadata_fields (List[str], optional): metadata column names to be saved.
            Defaults to _METADATA_FIELDS.
        workers (int, optional): number of workers to use in parallel when downloading the data.
            Defaults to 10.
        append_results (bool, optional): whether to save results for each post when downloading.
            Defaults to True.
        resume (bool, optional): option to resume downloading if the process breaks unexpectedly.
            Defaults to False.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        save_to: Union[str, Path],
        metadata_fields: List[str] = _METADATA_FIELDS,
        workers: int = 12,
        append_results: bool = True,
        resume: bool = False,
    ):
        self.df = pd.read_csv(dataset_path)
        self.save_to = save_to
        self.metadata_fields = metadata_fields
        self.workers = workers
        self.append_results = append_results
        self.resume = resume

        if not self.resume:
            self.cnt = 0
            tmp_dict = {field: [] for field in self.metadata_fields}
            pd.DataFrame(tmp_dict).to_csv(self.save_to, index=None)
        else:
            curr_df = pd.read_csv(self.save_to)
            self.cnt = len(curr_df)

    def filter_comments(
        self, comments: List[Tuple[str, str, int]], comment_ids: str
    ) -> List[str]:
        """Filters a list of comments based on comment ids.

        Args:
            comments (List[Tuple[str, str, int]]): list of comments
            comment_ids (str): string of list of comments

        Returns:
            List[str]: filtered list of comments
        """
        filtered_comments = []
        for com, id, _ in comments:
            if id in comment_ids:
                filtered_comments.append(com)
        return filtered_comments

    def flatten_nested_comments(
        self, comments_json: Dict[str, any], all_comms: Optional[List] = None
    ) -> List[Tuple[str, str, int]]:
        """Flattens a nested json dictionary with comments.

        Args:
            comments_json (Dict[str, any]): json dictionary with comments
            all_comms (Optional[List], optional): Defaults to None.

        Returns:
            List[Tuple[str, str, int]]: list of flattened comments
        """
        if all_comms is None:
            all_comms = []
        for comm in comments_json["children"]:
            if comm["kind"] == "t1":
                checks = [comm["data"]["distinguished"], comm["data"]["banned_by"]]
                if comm["data"]["author"] != "AutoModerator" and all(
                    check is None for check in checks
                ):
                    all_comms.append(
                        (
                            comm["data"]["body"],
                            comm["data"]["id"],
                            comm["data"]["depth"],
                        )
                    )
                    if len(comm["data"]["replies"]) > 0:
                        all_replies = self.flatten_nested_comments(
                            comm["data"]["replies"]["data"]
                        )
                        all_comms.extend(tuple(all_replies))
        return all_comms

    @retry(
        retry=retry_if_exception_type(HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_random(min=1, max=5),
    )
    def get_json_response(self, post_url: str) -> Dict:
        """Sends a GET request to get json response from the post url.
        Args:
            post_url (str): url of reddit post

        Raises:
            HTTPError: raises a HTTPError in case of 429 status code (too many requests)
        Returns:
            Dict: json response
        """
        json_response = None
        post_json_url = f"{post_url}.json"
        response = requests.get(post_json_url)
        if response.status_code == 200:
            try:
                json_response = json.loads(response.content)
            except Exception as e:
                logging.debug(
                    f"Failed to read json response {post_url} with error: {e}"
                )
        elif response.status_code in [403, 404]:
            logging.debug(f"Post not found, skipping {post_url}")
        elif response.status_code == 429:
            raise HTTPError(
                f"Couldn'd fetch post data for {post_url}, error code {response.status_code}"
            )
        return json_response

    def get_json_metadata(
        self, json_blobs: List[Dict[str, Any]], post_url: str
    ) -> Dict[str, Any]:
        """Gets json dictionary that corresponds to the post metadata (i.e. "t3").

        Args:
            json_blobs (List[Dict[str, Any]]): list of json dictionaries at different time points
            post_url (str): url of the reddit post

        Returns:
            Dict[str, Any]: json dictionary of post metadata
        """
        metadata_json = None
        try:
            metadata_json = [
                s
                for blob in json_blobs
                for s in blob["data"]["children"]
                if s["kind"] == "t3"
            ][0]["data"]
            if (
                metadata_json["removed_by_category"] is not None
                or metadata_json["selftext"] == "[removed]"
            ):
                logging.debug(f"{post_url} has been removed, skipping..")
                metadata_json = None
        except Exception as e:
            logging.debug(f"Failed to get json metadata for {post_url} with error {e}")
        return metadata_json

    def get_comments_metadata(
        self, json_blobs: List[Dict[str, Any]], post_url: str
    ) -> Dict[str, Any]:
        """Gets json dictionary that includes the highest number of comments.

        Args:
            json_blobs (List[Dict[str, Any]]): list of json dictionaries at different time points
            post_url (str): url of the reddit post

        Returns:
            Dict[str, Any]: json dictionary
        """
        comments_json = None
        try:
            max_len_blob_idx = np.argmax(
                [len(blob["data"]["children"]) for blob in json_blobs]
            )
            comments_json = json_blobs[max_len_blob_idx]
        except Exception as e:
            logging.debug(
                f"Failed to get comments metadata for {post_url} with error {e}"
            )
        return comments_json

    def get_comments_from_json(
        self, json_blob: Dict[str, Any], comment_ids: str
    ) -> List[str]:
        """Returns list of filtered comments from a json dictionary.

        Args:
            json_blob (Dict[str, Any]): json dictionary includind the comment thread
            comment_ids (str): filtered comment_ids

        Returns:
            List[str]: list of filtered comments
        """
        all_comments = self.flatten_nested_comments(json_blob["data"])
        comments = self.filter_comments(all_comments, comment_ids)
        comments = str(comments) if len(comments) > 0 else None
        return comments

    def update_csv(self, post_metadata: Dict[str, Any]):
        pd.DataFrame(post_metadata, index=[None]).to_csv(
            self.save_to, mode="a", header=False, index=None
        )

    def get_post_metadata(self, reddit_id: int) -> Dict[str, Any]:
        """Gets metadata for a post and appends it to a csv file if append_results is True.
        Args:
            reddit_id (int): reddit id

        Returns:
            Dict[str, Any]: dictionary with post metadata
        """
        post_metadata: Dict[str, Any] = {}
        row = self.df.loc[self.df.reddit_id == reddit_id]
        post_url = row.post_url.to_string(index=False)
        comment_ids = row.comment_ids.to_string(index=False)
        try:
            json_response = self.get_json_response(post_url)
        except Exception as e:
            logging.warning(
                f"Failed to get json for {post_url} with error {e}, try increasing the number of retries or try again later."
            )
            json_response = None
        metadata_json = self.get_json_metadata(json_response, post_url)

        if metadata_json is not None:
            post_metadata["reddit_id"] = reddit_id
            post_metadata["subreddit"] = metadata_json["subreddit"]
            post_metadata["video_url"] = row.video_url.to_string(index=False)
            post_metadata["title"] = metadata_json["title"]
            post_metadata["post_url"] = post_url
            post_metadata["image_preview_url"] = row.image_preview_url.to_string(
                index=False
            )
            comments_json = self.get_comments_metadata(json_response, post_url)
            post_metadata["comments"] = (
                self.get_comments_from_json(comments_json, comment_ids)
                if comments_json is not None
                else None
            )

            if self.append_results and post_metadata is not None:
                self.update_csv(post_metadata)

        return post_metadata

    def get_metadata(self) -> List[Dict[str, Any]]:
        """Downloads metadata in parallel including filtered comments and titles and saves them to a csv file.

        Returns:
            results: List[Dict[str, Any]]: DataFrame with results
        """
        results: List[Dict[str, Any]] = []
        ids = self.df[self.cnt :].reddit_id

        with tqdm(total=len(ids), desc=f"Downloading metadata") as pbar:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for response in executor.map(self.get_post_metadata, ids):
                    results.append(len(response) > 0)
                    pbar.update()
        logging.info(
            f"Done downloading metadata. Total downloaded posts: {len(results)}"
        )
        if not self.append_results:
            pd.DataFrame(results).to(self.save_to)
        return results


def main(args):
    downloader = VTCMetadataDownloader(
        dataset_path=args.csv,
        save_to=args.save_to,
        resume=args.resume,
        workers=args.num_workers,
    )
    downloader.get_metadata()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--csv",
        required=True,
        type=str,
        help="path to public VTC dataset csv (default: None)",
    )
    args.add_argument(
        "--save_to",
        required=True,
        type=str,
        help="csv path to save metadata to (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="resume downloading",
    )
    args.add_argument(
        "-nw",
        "--num_workers",
        default=12,
        type=int,
        help="number of workers to use when downloading files in parallel",
    )
    args = args.parse_args()
    main(args)
