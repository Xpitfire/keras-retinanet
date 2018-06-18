#!/bin/bash

# Start the second process
jupyter lab --NotebookApp.token= --port=9090 --no-browser --allow-root --ip=* &

# Start the first process
python app.py