import pandas as pd
import os
import json
from typing import List, Dict
from tiltify.config import Path


class DataLoader:

    def __init__(self, path: str = None) -> None:
        """
        Serves as a loader for policies. The data_path points by default to: "./data".
        Folder names are used for loading specific policies. It is assumed that every policy file within the
        respective folders is a json-file. If otherwise the loading has to be adjusted.

        Args:
            path (str, optional): _description_. Defaults to None.
        """
        if not path:
            self.data_path = Path.data_path
        else:
            self.data_path = path

    def get_json_data(self, data_folder: str = None) -> List[Dict]:
        if data_folder is None:
            data_folder = "official_policies"
        data_loading_path = os.path.join(self.data_path, data_folder)
        file_names = [file_name for file_name in os.listdir(data_loading_path) if file_name.endswith(".json")]
        loaded_policies = []
        for file_name in file_names:
            with open(os.path.join(data_loading_path, file_name), "r") as f:
                loaded_policies.append(json.load(f))
        return loaded_policies
