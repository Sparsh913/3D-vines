import open3d as o3d

# Load the source and target point clouds from .ply files
source_cloud = o3d.io.read_point_cloud("pointCloud_Top_Viewpoint_0.ply")
target_cloud = o3d.io.read_point_cloud("pointCloud_Top_Viewpoint_1.ply")

# Create a PCL Iterative Closest Point (ICP) object
icp = o3d.pipelines.registration.registration_icp(
    source_cloud, target_cloud, max_correspondence_distance=0.05,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint
()
)

# Get the transformation matrix
transformation_matrix = icp.transformation

# Print the transformation matrix
print("Transformation matrix:\n", transformation_matrix)

# Save the aligned source cloud
source_aligned = source_cloud.transform(transformation_matrix)
o3d.io.write_point_cloud("aligned_source_cloud.ply", source_aligned)

# Stitching Point clouds
stitched_cloud = source_aligned + target_cloud
o3d.io.write_point_cloud("stitched_cloud_top_0-1.ply", stitched_cloud)
