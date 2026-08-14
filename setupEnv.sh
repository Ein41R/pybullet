#!/bin/bash
# setupEnv.sh

# This script sets up the environment variables for the project.

f [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# ROS setup
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
fi

# Project-specific setup
mkdir -p data/logs data/models data/processed