import os, glob, csv, numpy as np, cv2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

def get_rosbag_options(path):
    return rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr')

def extract_all():
    bag_base_dir = os.path.expanduser('~/rosbag_data')
    output_dir = os.path.expanduser('~/dataset')
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'data.csv')

    bag_paths = sorted(glob.glob(os.path.join(bag_base_dir, 'auto_drive_dataset_*')))
    img_type = get_message('sensor_msgs/msg/Image')
    cmd_type = get_message('geometry_msgs/msg/Twist')
    total_frames = 0
    records = []

    for bag_path in bag_paths:
        storage_options, converter_options = get_rosbag_options(bag_path)
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        images, cmd_vels = [], []

        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == '/camera/image_raw':
                msg = deserialize_message(data, img_type)
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                if msg.encoding == 'rgb8': img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append((t, img))
            elif topic == '/cmd_vel':
                msg = deserialize_message(data, cmd_type)
                cmd_vels.append((t, msg.linear.x, msg.angular.z))

        cmd_times = np.array([c[0] for c in cmd_vels])
        for t_img, img in images:
            idx = np.argmin(np.abs(cmd_times - t_img))
            _, lin_x, ang_z = cmd_vels[idx]
            frame_filename = f"frame_{total_frames:06d}.jpg"
            cv2.imwrite(os.path.join(img_dir, frame_filename), img)
            records.append((f"images/{frame_filename}", round(lin_x, 4), round(ang_z, 4)))
            total_frames += 1

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'linear_x', 'angular_z'])
        writer.writerows(records)

if __name__ == '__main__':
    extract_all()
