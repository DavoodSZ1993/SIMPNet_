cmake_minimum_required(VERSION 3.1.3)
project(ur5e_gripper_moveit_config)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

#catkin_install_python(PROGRAMS
#  scripts/move_group.py scripts/collision_checking.py scripts/manipulation.py scripts/#real_time_listener.py scripts/test.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)
