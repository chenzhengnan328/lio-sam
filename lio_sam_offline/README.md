# lio_sam_offline

Offline mode of [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM). Remove ROS dependency.  
GPS factor is removed for single clip reconstruction.  
Loop closure is currently closed.  

## Dependency
opencv==4.5.1  
yaml-cpp==0.6.3  
gstam==4.0.2  
pcl==1.8.0  
ceres=2.0.0  
Eigen==3.3.9  

or you can use the docker below:  
docker.hobot.cc/imagesys/autopipeline:tool-kit_v1.3_cpu

## Usage  
### Compile  
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```
### Run
```bash
 ./build/liosam config/params_horizon.yaml FSD_Site_v1/Site_20211230_116_15472_40_06718/H3165_20211230_132320 0 0.05  
```
Usage: ./liosam config_path clip_path save_deskewed(1 to save, 0 to skip) save_map_resolution(meters, -1 to disable)


## Known Issues
If you suffer from cmake error  
Try following steps  
1. Uncomment Line#22-26 in CMakeLists.txt
2. add libtbb to the last line target_link_libraries

## TODO
1. Loop closure is currently disabled.  
