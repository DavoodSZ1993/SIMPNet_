# SIMPNet: Spatial-informed Motion Planning Network

<a href="TBD"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2408.12831"><strong>arXiv</strong></a>
  |
  <a href="https://x.com/davoodsz"><strong>Twitter</strong></a>
  
  
<a href="https://zh.engr.tamu.edu/people-2/">Davood Soleymanzadeh</a>,
<a href="https://engineering.tamu.edu/civil/profiles/liang-xiao.html">Xiao Liang</a>,
<a href="https://engineering.tamu.edu/mechanical/profiles/zheng-minghui.html">Minghui Zheng</a>.

**Robotics and Automation Letters (RA-L) (2025)**

<p align="center">
<img width="1000" src="./assets/02. SIMPNet-Structure.svg">
<br>
<em> Fig 1. Schematic of spatial-informed sampling heuristic within SIMPNet.</em>
</p>

This repository is the official implementation of SIMPNet: Spatial-informed Motion Planning Network.

<p>SIMPNet is a deep learning-based motion planning algorithm for robotic manipulators. It utliizes graph neural network and attention mechanism to generate informed samples within the framework of sampling-based motion planning algorithms. SIMPNet is designed to encode the kinematic and spatial strucutre of the robotic manipulator within the sampling heuristic for informed sampling. This repository contains the implementation, and evaluation scripts for SIMPNet.</p>

# Prerequisites
- Python 3.8
- Ubuntu 20.04.6 LTS (Focal Fossa)
- ROS Noetic
- MoveIt! 1

# Install
In a catkin workspace, clone the repo with submodules within the source folder and build the workspace.

```
cd catkin_ws/src
git clone --recursive https://github.com/DavoodSZ1993/SIMPNet.git
catkin build
```

# Training
All the models can be trained by accessing their corresponding folder. For instance, to train the SIMPNet for the simple environments, run:

```
cd simple_env/src/SIMPNet 
python3 main_train.py
```

# Evaluation
All the models can be evaluated as follows:

Launch UR5e manipulator for Robotiq gripper papcke

```
source ~/catkin_ws/src/setup.bash
roslaunch ur5e_gripper_moveit_config demo.launch
```

Source and run the planners:

```
source ~/catkin_ws/src/setup.bash
rosrun simple_env main_test_MPNN.py
```

# License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# Further Reading
- [MPNet GitHub Repo](https://github.com/anthonysimeonov/baxter_moveit_experiments).
- [MoveIt! Config for UR5e Manipulator with Robotiq Gripper.](https://roboticscasual.com/ros-tutorial-how-to-create-a-moveit-config-for-the-ur5-and-a-gripper/)

# Citation
If you find this codebase useful in your research, please cite:

```
@article{soleymanzadeh2024simpnet,
  title={SIMPNet: Spatial-Informed Motion Planning Network},
  author={Soleymanzadeh, Davood and Liang, Xiao and Zheng, Minghui},
  journal={arXiv preprint arXiv:2408.12831},
  year={2024}
}
```
