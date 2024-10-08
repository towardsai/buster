import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from buster.llm_utils import compute_embeddings_parallelized, get_openai_embedding

tqdm.pandas()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DocumentsManager(ABC):
    def __init__(self, required_columns: Optional[list[str]] = None):
        """
        Constructor for DocumentsManager class.

        Args:
            required_columns (Optional[list[str]]): A list of column names that are required for the dataframe to contain.
                                                     If None, no columns are enforced.
        """

        self.required_columns = required_columns

    def _check_required_columns(self, df: pd.DataFrame):
        """Each entry in the df is expected to have the columns in self.required_columns"""
        if not all(col in df.columns for col in self.required_columns):
            raise ValueError(f"DataFrame is missing one or more of {self.required_columns=}")

    def _checkpoint_csv(self, df, csv_filename: str, csv_overwrite: bool = True):
        """
        Saves DataFrame with embeddings to a CSV checkpoint.

        Args:
            df (pd.DataFrame): The DataFrame with embeddings.
            csv_filename (str): Path to save a copy of the dataframe with computed embeddings for later use.
            csv_overwrite (bool, optional): Whether to overwrite the file with a new file. Defaults to True.
        """
        import os

        if csv_overwrite:
            df.to_csv(csv_filename)
            logger.info(f"Saved DataFrame with embeddings to {csv_filename}")

        else:
            if os.path.exists(csv_filename):
                # append to existing file
                append_df = pd.read_csv(csv_filename)
                append_df = pd.concat([append_df, df])
            else:
                # will create the new file
                append_df = df.copy()
            append_df.to_csv(csv_filename)
            logger.info(f"Appending DataFrame embeddings to {csv_filename}")

    def add(
        self,
        df: pd.DataFrame,
        num_workers: int = 16,
        embedding_fn: Callable[[str], np.ndarray] = get_openai_embedding,
        sparse_embedding_fn: Callable[[str], dict[str, list[float]]] = None,
        csv_filename: Optional[str] = None,
        csv_overwrite: bool = True,
        **add_kwargs,
    ):
        """Write documents from a DataFrame into the DocumentManager store.

        This method adds documents from the provided DataFrame to the database. It performs the following steps:
        1. Checks if the required columns are present in the DataFrame.
        2. Computes embeddings for the 'content' column if they are not already present.
        3. Optionally saves the DataFrame with computed embeddings to a CSV checkpoint.
        4. Calls the '_add_documents' method to add documents with embeddings to the DocumentsManager.

        Args:
            df (pd.DataFrame): The DataFrame containing the documents to be added.
            num_workers (int, optional): The number of parallel workers to use for computing embeddings. Default is 32.
            embedding_fn (callable, optional): A function that computes embeddings for a given input string.
                Default is 'get_embedding_openai' which uses the text-embedding-ada-002 model.
            sparse_embedding_fn (callable, optional): A function that computes sparse embeddings for a given input string.
                Default is None. Only use if you want sparse embeddings.
            csv_filename (str, optional): Path to save a copy of the dataframe with computed embeddings for later use.
            csv_overwrite (bool, optional): Whether to overwrite the file with a new file. Defaults to True.
            **add_kwargs: Additional keyword arguments to be passed to the '_add_documents' method.
        """

        if self.required_columns is not None:
            self._check_required_columns(df)

        # Check if embeddings are present, computes them if not
        if "embedding" not in df.columns:
            embeddings = compute_embeddings_parallelized(df, embedding_fn=embedding_fn, num_workers=num_workers)
            df["embedding"] = pd.Series(embeddings)
        if "sparse_embedding" not in df.columns and sparse_embedding_fn is not None:
            sparse_embeddings = sparse_embedding_fn(df.content.to_list())
            df["sparse_embedding"] = pd.Series(sparse_embeddings)

        if csv_filename is not None:
            self._checkpoint_csv(df, csv_filename=csv_filename, csv_overwrite=csv_overwrite)

        self._add_documents(df, **add_kwargs)

    def batch_add(
        self,
        df: pd.DataFrame,
        batch_size: int = 3000,
        min_time_interval: int = 60,
        num_workers: int = 16,
        embedding_fn: Callable[[str], np.ndarray] = get_openai_embedding,
        sparse_embedding_fn: Callable[[str], dict[str, list[float]]] = None,
        csv_filename: Optional[str] = None,
        csv_overwrite: bool = False,
        **add_kwargs,
    ):
        """Adds DataFrame data to a DataManager instance in batches."""

        total_batches = (len(df) // batch_size) + 1

        logger.info(f"Adding {len(df)} documents with {batch_size=} for {total_batches=}")

        for batch_idx in range(total_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            start_time = time.time()

            # Calculate batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))

            # Instead of slicing, use .iloc to get a copy of the batch
            batch_df = df.iloc[start_idx:end_idx].copy()

            # Compute embeddings for the batch
            if "embedding" not in batch_df.columns:
                embeddings = compute_embeddings_parallelized(
                    batch_df, embedding_fn=embedding_fn, num_workers=num_workers
                )
                batch_df["embedding"] = embeddings
            if "sparse_embedding" not in batch_df.columns and sparse_embedding_fn is not None:
                sparse_embeddings = sparse_embedding_fn(batch_df.content.to_list())
                batch_df["sparse_embedding"] = sparse_embeddings

            # Save to CSV if filename is provided
            if csv_filename is not None:
                self._checkpoint_csv(batch_df, csv_filename=csv_filename, csv_overwrite=csv_overwrite)

            # Add the batch data using specified parameters
            self._add_documents(batch_df, **add_kwargs)

            elapsed_time = time.time() - start_time

            # Sleep to ensure the minimum time interval is maintained
            if batch_idx < total_batches - 1:
                sleep_time = max(0, min_time_interval - elapsed_time)
                if sleep_time > 0:
                    logger.info(f"Sleeping for {round(sleep_time)} seconds...")
                    time.sleep(sleep_time)

        logger.info("All batches processed.")

    @abstractmethod
    def _add_documents(self, df: pd.DataFrame, **add_kwargs):
        """Abstract method to be implemented by each inherited member.

        This method should handle the actual process of adding documents to the database.
        """
        ...
