cmake_minimum_required(VERSION 3.8)
project(smart_calibration_interfaces)

# rosidl's code generators run on the first python3 on PATH and need system Python packages
# (empy, lark, ...) that the isolated smart_calibration venv does not have.
if(DEFINED ENV{VIRTUAL_ENV})
  message(FATAL_ERROR
    "Build ${PROJECT_NAME} with the Python venv deactivated (active venv: $ENV{VIRTUAL_ENV}).\n"
    "Run 'deactivate', delete build/${PROJECT_NAME} (CMake caches the Python path) and "
    "build again. Activate the venv only for running the nodes.")
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# List every interface explicitly (e.g. "msg/BoardDetection.msg", "srv/Calibrate.srv").
set(interface_files
  "msg/BoardDetection.msg"
  "msg/CalibrationState.msg"
  "msg/FinalCalibration.msg"
  "msg/ToneHistogram.msg"
  "srv/Finish.srv"
  "srv/GetView.srv"
)

# rosidl_generate_interfaces() fails without any files, so it is only called once the first
# interface has been added above.
if(interface_files)
  rosidl_generate_interfaces(${PROJECT_NAME}
    ${interface_files}
    DEPENDENCIES std_msgs sensor_msgs geometry_msgs
  )
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
