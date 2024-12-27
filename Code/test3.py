

# Hata hesaplama
src_points = [pair[:2] for pair in point_map_example]
dst_points = [pair[2:] for pair in point_map_example]

error_opencv = back_projection_error(src_points, dst_points, H_opencv)
error_custom = back_projection_error(src_points, dst_points, H_custom)

print(f"OpenCV Back Projection Error: {error_opencv}")
print(f"Custom RANSAC Back Projection Error: {error_custom}")
