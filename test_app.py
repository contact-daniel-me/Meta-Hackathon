import sys
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

print("Calling POST /reset with empty body")
response = client.post("/reset", json={})
print("Status Code:", response.status_code)
print("Response:", response.json())

print("Calling POST /reset with difficulty")
response = client.post("/reset", json={"difficulty": "medium"})
print("Status Code:", response.status_code)
print("Response:", response.json())
