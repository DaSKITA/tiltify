import click
import requests
import json
from langdetect import detect

from tiltify.config import Path


def authorize(username, password, url):
    payload = json.dumps({
      "username": username,
      "password": password
    })
    return requests.request("POST", url + '/api/auth', headers={'Content-Type': 'application/json'}, data=payload).json()


def get_annotated_policies_json(token, url):
    return requests.request("GET", url + '/api/task/document_annotations', headers={'Authorization': token}).json()


def determine_policy_language(policy):
    policy_text = policy["document"]["text"]
    language = detect(policy_text)
    policy["document"]["language"] = language
    return policy


@click.command()
@click.option('--username', required=True)
@click.option('--password', required=True)
@click.option('--url', required=True)
def load_annotated_policies(username, password, url):
    jwt_token = authorize(username, password, url)
    annotated_policies = get_annotated_policies_json(jwt_token, url)
    for entry in annotated_policies:
        entry = determine_policy_language(entry)
        path = Path.annotated_policy_path + '/' + entry["document"]["document_name"] + '.json'
        with open(path, 'w') as file:
            json.dump(entry, file)


if __name__ == "__main__":
    load_annotated_policies()
    # from glob import glob

    # policy_paths = "/home/gebauer/Desktop/repo/tiltify/data/annotated_policies/*.json"
    # for policy_path in glob(policy_paths):
    #     with open(policy_path, "r") as p:
    #         policy = json.load(p)
    #     policy = determine_policy_language(policy)
    #     with open(policy_path, "w") as p:
    #         json.dump(policy, p)
