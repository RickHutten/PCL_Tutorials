#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/visualization/cloud_viewer.h>

int main ()
{
    // load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile ("../models/milk.pcd", *cloud);

    std::cout << "File has " << cloud->points.size() << " points" << std::endl;

    // estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (0.01);

    ne.compute(*normals);

    // Create the PFH estimation class, and pass the input dataset+normals to it
    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
    pfh.setInputCloud (cloud);
    pfh.setInputNormals (normals);
    // alternatively, if cloud is of tpe PointNormal, do pfh.setInputNormals (cloud);
    pfh.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    pfh.setRadiusSearch (0.02);

    for (int i = 0; i < normals->points.size(); i++)
    {
        if (!pcl::isFinite<pcl::Normal>(normals->points[i]))
        {
            PCL_WARN("normals[%d] is not finite\n", i);
        }
    }

    // Compute the features, takes forever
    pfh.compute (*pfhs);

    return 0;
}