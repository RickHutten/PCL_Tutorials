#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

int main(int argc, char **argv)
{
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());

    // Fill in the cloud data
    pcl::PCDReader reader;

    // Replace the path below with the path where you saved your file
    reader.read("../models/table_scene_lms400.pcd", *cloud);

    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
              << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
              << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

    pcl::PCDWriter writer;
    writer.write("../models/output/table_scene_lms400_downsampled.pcd", *cloud_filtered,
                 Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), false);

    return (0);
}