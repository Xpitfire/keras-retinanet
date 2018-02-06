#!/bin/bash

# Start the second process
jupyter-notebook --port=9090 --no-browser --allow-root --ip=* &

# Start the first process
python3 app.py