import requests

class CatDetectorClient:
    def __init__(self, url):
        self.url = url

    def send_file(self, file_path):
        # Open the file in binary mode
        with open(file_path, 'rb') as f:
            # Define the file payload to send
            files = {'file': (file_path, f, 'audio/wav')}
            # Make the POST request to the server
            response = requests.post(self.url, files=files)
            return response.json()