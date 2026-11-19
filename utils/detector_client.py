from typing import Dict, Optional

import requests


class CatDetectorClient:
    """Sends recordings to the cat detection server."""

    def __init__(self, url: str, timeout_s: float = 10.0):
        self.url = url
        self.timeout_s = timeout_s

    def send_file(self, file_path: str,
                  metadata: Optional[Dict[str, str]] = None) -> Optional[dict]:
        """Send a WAV file (plus optional form fields) and return the server's JSON response.

        Returns None if the server is unavailable.
        """
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path, f, 'audio/wav')}
                response = requests.post(self.url, files=files, data=metadata or {},
                                         timeout=self.timeout_s)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            # The server runs on a PC that may be off; never crash the Pi app over it.
            print(f"Detection server unavailable ({self.url}): {exc}")
            return None
