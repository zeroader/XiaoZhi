import requests

url = "http://127.0.0.1:8291/lamp"

data = {
    "action": "set_power",
    "value": True
}

response = requests.post(url, json=data, timeout=5)

print(response.json())