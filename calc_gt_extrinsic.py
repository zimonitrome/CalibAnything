import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(q):
    rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy expects (x, y, z, w)
    rotation_matrix = rotation.as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    return transformation_matrix

def translation_to_matrix(translation):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

def combine_rotation_translation(quaternion, translation):
    rotation_matrix = quaternion_to_matrix(quaternion)
    translation_matrix = translation_to_matrix(translation)
    return rotation_matrix @ translation_matrix

camera_quaternion = [0.490061, -0.499996, 0.513669, -0.495972] # (w, x, y, z)
camera_translation = [1.899705, 0.098615, 1.687573] # (x, y, z)
base_to_camera_matrix = combine_rotation_translation(camera_quaternion, camera_translation)

lidar_quaternion = [0.000169, -0.006562, 0.003146, -0.999974] # (w, x, y, z)
lidar_translation = [1.788764, 0.604874, 1.698761] # (x, y, z)
base_to_lidar_matrix = combine_rotation_translation(lidar_quaternion, lidar_translation)

camera_to_base_matrix = np.linalg.inv(base_to_camera_matrix)

lidar_to_camera_matrix = camera_to_base_matrix @ base_to_lidar_matrix

np.set_printoptions(
    suppress=True,
    formatter={'float': lambda x: f"{x: 0.5f}" if x >= 0 else f"{x:0.5f}"}
)
print(np.array2string(lidar_to_camera_matrix, separator=', '))
