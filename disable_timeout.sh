#!/bin/bash


# Disable the jetson's GPU timetout after ~5 sec of kernel launch
# To be executed as sudo:
#    sudo ./disable_timeout.sh

echo N > /sys/kernel/debug/gpu.0/timeouts_enabled
timer=`cat /sys/kernel/debug/gpu.0/timeouts_enabled`
echo "Timeout disabled" 
