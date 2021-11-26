from tiltify.config import Path
import json
import os


file_path = os.path.join(Path.policy_path, "basf.json")

with open(os.path.join(Path.policy_path, "basf.json")) as f:
    json_file = json.load(f)

print(json_file["text"])


