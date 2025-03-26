import requests
import json

url = "http://0.0.0.0:8000/query"
headers = {"Content-Type": "application/json"}
data = {"query": "What is an operating system?"}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    formatted_response = json.dumps(response.json(), indent=4)
    print("Response:", formatted_response)
else:
    print("Failed to get a response. Status code:", response.status_code)