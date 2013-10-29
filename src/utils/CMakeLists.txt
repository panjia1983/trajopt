include_directories(
  ${Eigen_INCLUDE_DIRS}
)
add_library(utils stl_to_string.cpp clock.cpp config.cpp logging.cpp timer.cpp random_utils.cpp)
target_link_libraries(utils ${Boost_PROGRAM_OPTIONS_LIBRARY})
