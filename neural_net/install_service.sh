#!/bin/bash

# Get the current username
USER_NAME=$(whoami)

# Get the current script path
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

SERVICE_NAME="cat_detector_service"
SERVICE_PATH="$SCRIPT_PATH/start_server.sh"
SERVICE_WORKING_DIRECTORY="$SCRIPT_PATH"

# Create a systemd service file
echo "[Unit]
Description=Image Segmentation Service

[Service]
ExecStart=$SERVICE_PATH
Restart=always
User=$USER_NAME
Group=$USER_NAME
Environment=PATH=/usr/bin:/usr/local/bin
Environment=NODE_ENV=production
WorkingDirectory=$SERVICE_WORKING_DIRECTORY

[Install]
WantedBy=multi-user.target
" | sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null

# Make script executable
chmod +x $SERVICE_PATH

# Enable and start the service
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME
