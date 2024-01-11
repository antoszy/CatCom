# Use the base image with TensorFlow, GPU support, and Jupyter Notebook
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set the working directory
WORKDIR /notebooks

# Install additional Python packages using pip
RUN pip install pandas tensorflow_io tensorflow_hub

# Expose the Jupyter Notebook port
EXPOSE 8080

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--allow-root"]
