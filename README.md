# isaac_sim_test
This repo contains serval script to test isaac sim interfaces

## ISAAC SIM TO ROS2
```bash
# Terminal 1
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mingqian/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/exts/isaacsim.ros2.bridge/jazzy/lib
export ROS_DOMAIN_ID=0
alias isaac_sim='/home/mingqian/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/isaac-sim.selector.sh'
alias isaac_py='/home/mingqian/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh'

isaac_py /home/mingqian/Desktop/IsaacSim_test/ros2_depth.py
```

```bash
# Terminal 2
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
rviz2
```
