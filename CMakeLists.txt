cmake_minimum_required(VERSION 3.1)
project(pcl-tutorial)

## C++11 Flags
##
add_compile_options(-std=c++11)
add_compile_options(-pthread)
add_compile_options(-Werror=return-type)                   # error on missing return type
add_compile_options(-Werror=format)                        # error on wrong printf formats
add_compile_options(-Werror=parentheses)                   # error when using ambiguous syntax

set(STRICT_GCC_FLAGS "\
    -Wunused-but-set-parameter \
    -Wunused-but-set-variable \
    -Wunused-function \
    -Wunused-variable \
    -Wuninitialized \
    -Wmaybe-uninitialized \
    -Wno-endif-labels \
    -Wmissing-field-initializers \
    -Wunused-parameter
")

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#Searching CUDA
find_package(CUDA)

#Include the FindCUDA script
include(FindCUDA)

include_directories(utils)

### Executables ###

# Basic
add_executable(CloudTransformation basic/CloudTransformation.cpp)
target_link_libraries(CloudTransformation ${PCL_LIBRARIES})

add_executable(NearestNeighborSearch basic/NearestNeighborSearch.cpp)
target_link_libraries(NearestNeighborSearch ${PCL_LIBRARIES})

# Filters
add_executable(PassThrough filter/PassThrough.cpp)
target_link_libraries(PassThrough ${PCL_LIBRARIES})

add_executable(VoxelGrid filter/VoxelGrid.cpp)
target_link_libraries(VoxelGrid ${PCL_LIBRARIES})

add_executable(StatisticalOutlierRemoval filter/StatisticalOutlierRemoval.cpp)
target_link_libraries(StatisticalOutlierRemoval ${PCL_LIBRARIES})

add_executable(ExtractingIndices filter/ExtractingIndices.cpp)
target_link_libraries(ExtractingIndices ${PCL_LIBRARIES})

# Features
add_executable(NormalEstimationUsingIntegralImages features/NormalEstimationUsingIntegralImages.cpp)
target_link_libraries(NormalEstimationUsingIntegralImages ${PCL_LIBRARIES})

add_executable(EstimatingSurfaceNormals features/EstimatingSurfaceNormals.cpp)
target_link_libraries(EstimatingSurfaceNormals ${PCL_LIBRARIES})

add_executable(PointFeatureHistograms features/PointFeatureHistograms.cpp)
target_link_libraries(PointFeatureHistograms ${PCL_LIBRARIES})

add_executable(FastPointFeatureHistograms features/FastPointFeatureHistograms.cpp)
target_link_libraries(FastPointFeatureHistograms ${PCL_LIBRARIES})

add_executable(NARFFeaturesFromRangeImage features/NARFFeaturesFromRangeImage.cpp)
target_link_libraries(NARFFeaturesFromRangeImage ${PCL_LIBRARIES})

add_executable(RotationalProjectionStatistics features/RotationalProjectionStatistics.cpp)
target_link_libraries(RotationalProjectionStatistics ${PCL_LIBRARIES})

# Segmentation
add_executable(PlaneModelSegmentation segmentation/PlaneModelSegmentation.cpp)
target_link_libraries(PlaneModelSegmentation ${PCL_LIBRARIES})

add_executable(CylinderModelSegmentation segmentation/CylinderModelSegmentation.cpp)
target_link_libraries(CylinderModelSegmentation ${PCL_LIBRARIES})

add_executable(EuclideanClusterExtraction segmentation/EuclideanClusterExtraction.cpp)
target_link_libraries(EuclideanClusterExtraction ${PCL_LIBRARIES})

# Surface
add_executable(Resampling surface/Resampling.cpp)
target_link_libraries(Resampling ${PCL_LIBRARIES})

add_executable(ConcaveConvexHull2D surface/ConcaveConvexHull2D.cpp)
target_link_libraries(ConcaveConvexHull2D ${PCL_LIBRARIES})

add_executable(Triangulation surface/Triangulation.cpp)
target_link_libraries(Triangulation ${PCL_LIBRARIES})

# Registration
add_executable(IterativeClosestPoint registration/IterativeClosestPoint.cpp)
target_link_libraries(IterativeClosestPoint ${PCL_LIBRARIES})

add_executable(ICPMultipleClouds registration/ICPMultipleClouds.cpp)
target_link_libraries(ICPMultipleClouds ${PCL_LIBRARIES})

add_executable(NormalDistributionTransform registration/NormalDistributionTransform.cpp)
target_link_libraries(NormalDistributionTransform ${PCL_LIBRARIES})

add_executable(PoseEstimation registration/PoseEstimation.cpp)
target_link_libraries(PoseEstimation ${PCL_LIBRARIES})

# Recognition
add_executable(CorrespondenceGrouping recognition/CorrespondenceGrouping.cpp)
target_link_libraries(CorrespondenceGrouping ${PCL_LIBRARIES})

add_executable(ImplicitShapeModel recognition/ImplicitShapeModel.cpp)
target_link_libraries(ImplicitShapeModel ${PCL_LIBRARIES})

# GPU
add_executable(GPU-FastPointFeatureHistograms gpu/FastPointFeatureHistograms.cpp)
target_link_libraries(GPU-FastPointFeatureHistograms ${PCL_LIBRARIES})

add_executable(GPU-Segmentation gpu/Segmentation.cpp)
target_link_libraries(GPU-Segmentation ${PCL_LIBRARIES})

add_executable(GPU-PoseEstimation gpu/PoseEstimation.cpp)
target_link_libraries(GPU-PoseEstimation ${PCL_LIBRARIES})
