#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#include "timer.hpp"

/**
 * Checks if one pointcloud is just a rigid transformation of another by minimizing
 * the distances between the points of two pointclouds and rigidly transforming them.
 *
 * Accuracy goes to shit if number of points grows above 1000+ (sometimes even just at 100+)
 */

int main(int argc, char **argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud_source) == -1)
    {
        PCL_ERROR ("Couldn't read 1st file \n");
        return (1);
    }

    // Create transformed cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *cloud_transformed) == -1)
    {
        PCL_ERROR ("Couldn't read 2nd file \n");
        return (1);
    }
    
    ElapseTimer t;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_source);
    icp.setInputTarget(cloud_transformed);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    std::cout << "\nAligning took: " << t.elapsed() << std::endl;

    std::cout << "Has converged: " << icp.hasConverged() << " Fitness score: " << icp.getFitnessScore() << std::endl;
    std::cout << "Calculated transform:\n" << icp.getFinalTransformation() << std::endl;

    std::stringstream ss;
    ss << "/home/intern2/Programs/ros_test_workspace/transformed.pcd";
    pcl::io::savePCDFileASCII(ss.str(), Final);

    return (0);
}