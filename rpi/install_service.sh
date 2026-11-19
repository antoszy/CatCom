#!/bin/bash
# Install a systemd service that starts the Kotofon app (detect_cat.py) at boot,
# already detecting (--start-running). Run on the Raspberry Pi as the app user.
set -euo pipefail

SERVICE_NAME="kotofon"
USER_NAME="$(whoami)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The app's dependencies (RPi.GPIO, pyaudio, scikit-learn, requests, ...) are
# installed for the system interpreter, not in a venv.
PYTHON="${KOTOFON_PYTHON:-/usr/bin/python3}"

echo "[Unit]
Description=Kotofon cat meow detector
# Mic/speaker HAT driver must be configured before the audio stream is opened.
After=seeed-voicecard.service sound.target network-online.target
Wants=seeed-voicecard.service network-online.target

[Service]
ExecStart=$PYTHON -u $REPO_DIR/detect_cat.py --start-running
WorkingDirectory=$REPO_DIR
User=$USER_NAME
Group=$USER_NAME
SupplementaryGroups=audio gpio
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
" | sudo tee "/etc/systemd/system/$SERVICE_NAME.service" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
echo "Installed. Logs: journalctl -u $SERVICE_NAME -f"
