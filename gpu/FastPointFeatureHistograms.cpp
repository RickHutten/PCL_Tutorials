#include <pcl/common/time.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h> // Multi-core/threading using OpenMP
#include <pcl/visualization/cloud_viewer.h>

// GPU includes
#include <pcl/gpu/features/features.hpp>
#include <pcl/cuda/time_cpu.h>
#include <pcl/cuda/time_gpu.h>

/*
{
    pcl::ScopeTime t("Name");
}
*/

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <cloud.pcd>" << std::endl;
        exit(1);
    }

    // load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile (argv[1], *cloud);
    std::cout << "Cloud has " << cloud->points.size() << " points" << std::endl;

    // estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.05);
    ne.compute(*normals);

    /// CPU - NO OMP
    {
        pcl::ScopeTime t("CPU FPFH KdTree");

        pcl::FPFHEstimation <pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        //pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh; // Multi-core/threading using OpenMP
        fpfh.setInputCloud(cloud);
        fpfh.setInputNormals(normals);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
        fpfh.setSearchMethod(tree2);

        // Output datasets
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

        // Use all neighbors in a sphere of radius 10cm
        fpfh.setRadiusSearch(0.1);

        // Compute the features
        fpfh.compute(*fpfhs);
    }

    /// CPU - OMP
    {
        pcl::cuda::ScopeTimeCPU t("CPU FPFH-OMP KdTree");

        pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setInputCloud(cloud);
        fpfh.setInputNormals(normals);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
        fpfh.setSearchMethod(tree2);

        // Output datasets
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

        // Use all neighbors in a sphere of radius 10cm
        fpfh.setRadiusSearch(0.1);

        // Compute the features
        fpfh.compute(*fpfhs);
    }
    /// CPU - OMP - OCTREE
    {
        pcl::ScopeTime t("CPU FPFH-OMP Octree");

        pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setInputCloud(cloud);
        fpfh.setInputNormals(normals);
        pcl::search::Octree<pcl::PointXYZ>::Ptr oct;
        fpfh.setSearchMethod(oct);

        // Output datasets
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

        // Use all neighbors in a sphere of radius 10cm
        fpfh.setRadiusSearch(0.1);

        // Compute the features
        fpfh.compute(*fpfhs);
    }
    /// GPU
    {
        pcl::cuda::ScopeTimeGPU t("GPU FPFH Octree");

        pcl::gpu::FPFHEstimation::PointCloud cloud_gpu;
        cloud_gpu.upload(cloud->points);

        std::vector<pcl::PointXYZ> normals_for_gpu(normals->points.size());

        struct Normal2PointXYZ
        {
            pcl::PointXYZ operator()(const pcl::Normal &n) const
            {
                pcl::PointXYZ xyz;
                xyz.x = n.normal[0];
                xyz.y = n.normal[1];
                xyz.z = n.normal[2];
                return xyz;
            }
        };
        Normal2PointXYZ s;

        std::transform(normals->points.begin(), normals->points.end(), normals_for_gpu.begin(), s);

        pcl::gpu::FPFHEstimation::Normals normals_gpu;
        normals_gpu.upload(normals_for_gpu);

        pcl::gpu::FPFHEstimation fe_gpu;
        fe_gpu.setInputCloud(cloud_gpu);
        fe_gpu.setInputNormals(normals_gpu);
        fe_gpu.setRadiusSearch(0.1, 10000); // 10000 max points

        pcl::gpu::DeviceArray2D<pcl::FPFHSignature33> fpfhs_gpu;

        {
            pcl::ScopeTime t2("GPU FPFH Octree - Computation part");
            fe_gpu.compute(fpfhs_gpu);
        }
    }

    return 0;
}
