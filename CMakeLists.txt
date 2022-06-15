cmake_minimum_required(VERSION 3.5)
project(lio_sam_offline)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17")
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations") # disable PCL deprecation warning

find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)

set(OpenMP_CXX_FLAGS "/usr/lib/x86_64-linux-gnu/libSM.so")

# set(PCL_LIBRARY_DIRS /home/users/nemo.liu/data_pipeline/third/lib)
# set(PCL_INCLUDE_DIRS /home/users/nemo.liu/data_pipeline/third/include/pcl-1.8)

# set(Boost_INCLUDE_DIRS /home/users/nemo.liu/data_pipeline/third/local/include)
# set(Boost_LIBRARY_DIRS /home/users/nemo.liu/data_pipeline/third/local/lib)
# set(OpenCV_LIBRARY_DIRS /home/users/nemo.liu/data_pipeline/third/lib64)

# set(GTSAM_LIBRARY_DIRS /home/users/nemo.liu/data_pipeline/third/lib)
# set(GTSAM_INCLUDE_DIR /home/users/nemo.liu/data_pipeline/third/include)

# # Turn this block on if you encounter following error:
# #/usr/lib/x86_64-linux-gnu/libtbb.so.2:error adding symbols: DSO missing from command line
# set(LIB_OPENCV_IMGPROC_DIR /usr/lib/x86_64-linux-gnu)
# add_library(libtbb SHARED IMPORTED)
# set_target_properties(libtbb PROPERTIES IMPORTED_LOCATION ${LIB_OPENCV_IMGPROC_DIR}/libtbb.so.2)



# include directories
include_directories(
	include
  ${PCL_INCLUDE_DIRS}
	${yaml-cpp_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# link directories
link_directories(	
  include
	${PCL_LIBRARY_DIRS}
  yaml-cpp
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)


###########
## Build ##
###########


# Utility library
add_library(utility  src/utility.cpp)
target_link_libraries(utility yaml-cpp)

# # Range Image Projection
add_library(imageProjection src/imageProjection.cpp)
target_link_libraries(imageProjection  utility ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Feature Association
add_library(featureExtraction src/featureExtraction.cpp)
target_link_libraries(featureExtraction utility ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# # Mapping Optimization
add_library(mapOptmization src/mapOptmization.cpp)
target_compile_options(mapOptmization PRIVATE ${OpenMP_CXX_FLAGS}) #
target_link_libraries(mapOptmization  ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam) #${OpenMP_CXX_FLAGS}

# IMU Preintegration
add_library(imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(imuPreintegration  ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

# auto DatasetReader
add_library(autoDatasetReader src/autoDatasetReader.cpp)
target_link_libraries(autoDatasetReader ${PCL_LIBRARIES})


# main function
add_executable(liosam src/main.cpp)
target_link_libraries(liosam autoDatasetReader featureExtraction imageProjection mapOptmization imuPreintegration)