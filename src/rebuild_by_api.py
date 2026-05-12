import requests

headers = {"X-API-Key": "your-super-secret-key-change-me-in-production"}
response = requests.post("http://localhost:8000/rebuild", headers=headers)
print(response.json())