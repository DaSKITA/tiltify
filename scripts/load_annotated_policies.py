import click
import requests
import json

from tiltify.config import Path


def authorize(username, password, url):
    payload = json.dumps({
      "username": username,
      "password": password
    })
    return requests.request("POST", url + '/api/auth', headers={'Content-Type': 'application/json'}, data=payload).json()


def get_annotated_policies_json(token, url):
    return requests.request("GET", url + '/api/task/document_annotations', headers={'Authorization': token}).json()


@click.command()
@click.option('--username', required=True)
@click.option('--password', required=True)
@click.option('--url', required=True)
def load_annotated_policies(username, password, url):
    jwt_token = authorize(username, password, url)
    annotated_policies = get_annotated_policies_json(jwt_token, url)
    for entry in annotated_policies:
        path = Path.annotated_policy_path + '/' + entry["document"]["document_name"] + '.json'
        with open(path, 'w') as file:
            json.dump(entry, file)


if __name__ == "__main__":
    load_annotated_policies()