#!/bin/bash


# setting up environment
# run from project root

#change power policy of pc (only for me)
echo max_performance | sudo tee /sys/class/scsi_host/host*/link_power_management_policy

# python venv setup
source bin/activate

###
source /opt/ros/jazzy/setup.bash
source assets/Universal_Robots/install/setup.bash