#!/bin/bash

# Run a Docker container with TensorFlow GPU and Jupyter Notebook.
# Map port 8080 from the host to port 8080 inside the container.
# Mount the current directory (.) to the /notebooks directory inside the container.
# docker run -it --rm -p 8080:8080 -v "$(pwd)":/notebooks tensorall \
#     bash 

# docker run --gpus all -it --rm -p 8081:8081 -v "$(pwd)":/notebooks -v /data:/data tensorall  \
#     bash -c "uvicorn cat_detection_server:app --port 8081 --reload --ssl-keyfile=key.pem --ssl-certfile=cert.pem"

docker run -it --rm -p 8981:8981 -v "$(pwd)":/notebooks -v /data:/data tensorall  \
    bash -c "uvicorn cat_detection_server:app --port 8981 --host 0.0.0.0 --reload"