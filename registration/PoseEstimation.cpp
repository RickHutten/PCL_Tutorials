#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT, PointNT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

// Align a rigid object to a scene with clutter and occlusions
int main(int argc, char **argv)
{
    if (argc != 9) {
        std::cerr << "Give two pointcloud .pcd files as argument to align!" << std::endl;
        exit(1);
    }

    /*
    align.setMaximumIterations(50000); // Number of RANSAC iterations
    align.setNumberOfSamples(number_of_samples); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(correspondance_randomness); // Number of nearest features to use
    align.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(2.5f * leaf); // Inlier threshold
    align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis
     */

    float leaf = 0.5;
    int number_of_samples = 3; // Number of points to sample for generating/prerejecting a pose
    int correspondance_randomness = 5; // Number of nearest features to use
    float similarity_threshold = 0.9; // Polygonal edge length similarity threshold
    float max_correspondance_threshold = 2.5; // Inlier threshold
    float inlier_fraction = 0.25; // Required inlier fraction for accepting a pose hypothesis

    leaf = strtof(argv[3], NULL);
    number_of_samples = atoi(argv[4]);
    correspondance_randomness = atoi(argv[5]);
    similarity_threshold = strtof(argv[6], NULL);
    max_correspondance_threshold = strtof(argv[7], NULL);
    inlier_fraction = strtof(argv[8], NULL);


    // Point clouds
    PointCloudT::Ptr object(new PointCloudT);
    PointCloudT::Ptr object_aligned(new PointCloudT);
    PointCloudT::Ptr scene(new PointCloudT);
    FeatureCloudT::Ptr object_features(new FeatureCloudT);
    FeatureCloudT::Ptr scene_features(new FeatureCloudT);

    // Load object and scene
    pcl::console::print_highlight("Loading point clouds...\n");
    pcl::io::loadPCDFile<PointNT>(argv[1], *object);
    pcl::io::loadPCDFile<PointNT>(argv[2], *scene);

    // Downsample
    pcl::console::print_highlight("Downsampling...\n");
    pcl::ApproximateVoxelGrid<PointNT> vg;
    vg.setLeafSize(leaf, leaf, leaf);
    vg.setInputCloud(object);
    vg.filter(*object);
    vg.setInputCloud(scene);
    vg.filter(*scene);

    // Estimate normals for scene
    pcl::console::print_highlight("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<PointNT, PointNT> nest;
    nest.setRadiusSearch(1);
    nest.setInputCloud(scene);
    nest.compute(*scene);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch(1); // 0.025
    fest.setInputCloud(object);
    fest.setInputNormals(object);
    fest.compute(*object_features);
    fest.setInputCloud(scene);
    fest.setInputNormals(scene);
    fest.compute(*scene_features);

    // Perform alignment
    pcl::console::print_highlight("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align;
    align.setInputSource(object);
    align.setSourceFeatures(object_features);
    align.setInputTarget(scene);
    align.setTargetFeatures(scene_features);
    align.setMaximumIterations(50000); // Number of RANSAC iterations
    align.setNumberOfSamples(number_of_samples); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(correspondance_randomness); // Number of nearest features to use
    align.setSimilarityThreshold(similarity_threshold); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(max_correspondance_threshold * leaf); // Inlier threshold
    align.setInlierFraction(inlier_fraction); // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align(*object_aligned);
    }

    std::cout << "Result: " << align.getFitnessScore() << " " << align.hasConverged() << std::endl;

    if (align.hasConverged())
    {
        // Print results
        printf("\n");
        Eigen::Matrix4f transformation = align.getFinalTransformation();
        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1),
                                 transformation(0, 2));
        pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1),
                                 transformation(1, 2));
        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1),
                                 transformation(2, 2));
        pcl::console::print_info("\n");
        pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3),
                                 transformation(2, 3));
        pcl::console::print_info("\n");
        pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), object->size());

        // Show alignment
        pcl::visualization::PCLVisualizer visu("Alignment");
        visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0), "scene");
        visu.addPointCloud(object_aligned, ColorHandlerT(object_aligned, 0.0, 0.0, 255.0), "object_aligned");
        visu.spin();
    } else
    {
        pcl::console::print_error("Alignment failed!\n");
        return (1);
    }

    return (0);
}