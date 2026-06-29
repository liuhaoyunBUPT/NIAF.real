#!/usr/bin/env python3
import matplotlib.pyplot as plt
# Set backend to Agg for headless environments
plt.switch_backend('Agg')
import numpy as np
import sys
import os
import csv
import argparse
import time
import math

# Global list to store data in memory before saving
# Each record is a dictionary: {'time': t, 'pos': [...], 'vel': [...]}
data_records = []
master_data_records = []  # For master joint commands
start_time = None

def callback(msg):
    global start_time
    # Use msg timestamp if available and non-zero, else system time
    current_time_abs = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) / 1e9
    if current_time_abs == 0:
        current_time_abs = time.time()

    if start_time is None:
        start_time = current_time_abs

    rel_time = current_time_abs - start_time
    
    # Store position and velocity
    # msg.position is joint angles
    # msg.velocity is joint velocities (qvel)
    # Note: msg.velocity might be an empty tuple (), so we always convert to list
    # and check length later in save_csv
    rec = {
        'time': rel_time,
        'pos': list(msg.position),
        'vel': list(msg.velocity)  # Direct conversion, like collect_data.py does
    }
    data_records.append(rec)

def master_callback(msg):
    global start_time
    # Use msg timestamp if available and non-zero, else system time
    current_time_abs = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) / 1e9
    if current_time_abs == 0:
        current_time_abs = time.time()

    if start_time is None:
        start_time = current_time_abs

    rel_time = current_time_abs - start_time
    
    rec = {
        'time': rel_time,
        'pos': list(msg.position),
        'vel': list(msg.velocity) if msg.velocity else []
    }
    master_data_records.append(rec)

def calculate_derivative(times, values):
    """Calculates discrete derivative (p(t) - p(t-1)) / dt"""
    derivs = []
    if len(times) < 2 or not values:
        if values:
            return [[0.0] * len(values[0])] * len(values)
        return []
    
    num_joints = len(values[0])
    # Initial derivative is 0 or assume static
    derivs.append([0.0] * num_joints)
    
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        row = []
        for j in range(num_joints):
            if dt > 1e-6: # Avoid division by zero
                # diff = (curr - prev) / dt
                val = (values[i][j] - values[i-1][j]) / dt
            else:
                val = 0.0
            row.append(val)
        derivs.append(row)
    
    return derivs

def save_single_csv(filename, times, data, header):
    print(f"Saving {len(data)} records to {filename}...")
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t, row in zip(times, data):
                writer.writerow([t] + row)
        print(f"Saved successfully to {filename}.")
    except Exception as e:
        print(f"Error saving CSV {filename}: {e}")

def save_position_only_csv(filename, data_records):
    """Save only position data (for master commands)"""
    if not data_records:
        print("No data to save.")
        return
        
    times = [r['time'] for r in data_records]
    positions = [r['pos'] for r in data_records]
    
    # Determine header
    num_joints = len(positions[0])
    header = ["time"] + [f"joint{i}" for i in range(num_joints)]
    
    # Save only position
    save_single_csv(filename, times, positions, header)

def save_csv(filename):
    if not data_records:
        print("No data to save.")
        return
        
    times = [r['time'] for r in data_records]
    positions = [r['pos'] for r in data_records]
    
    # Determine header
    num_joints = len(positions[0])
    header = ["time"] + [f"joint{i}" for i in range(num_joints)]
    
    # 1. Save Position (Original)
    save_single_csv(filename, times, positions, header)
    
    root, ext = os.path.splitext(filename)

    # 2. Handle Velocity
    # Check if ROS provided velocity
    has_ros_vel = False
    ros_velocities = [r['vel'] for r in data_records]
    
    # Check if we have valid velocity data (matching dimensions)
    # Based on collect_data.py, 'qvel' comes directly from msg.velocity
    # We accept the data even if it is all zeros (static robot)
    if ros_velocities and len(ros_velocities) > 0 and len(ros_velocities[0]) == num_joints:
        has_ros_vel = True
    
    final_velocities = []

    if has_ros_vel:
        print(f"Velocity Source: [ROS Topic] (Saving to *_vel{ext})")
        final_velocities = ros_velocities
    else:
        print(f"Velocity Source: [Calculated Diff] (ROS velocity missing/empty, saving to *_vel{ext})")
        final_velocities = calculate_derivative(times, positions)

    vel_filename = f"{root}_vel{ext}"
    save_single_csv(vel_filename, times, final_velocities, header)

    # 3. Handle Acceleration
    # Standard JointState does not have acceleration. We default to calculation.
    # Logic: If we read velocity from ROS, we diff that. If we calculated velocity, we diff that (2nd derivative of pos).
    print(f"Acceleration Source: [Calculated Diff] (Saving to *_acc{ext})")
    final_accelerations = calculate_derivative(times, final_velocities)
    
    acc_filename = f"{root}_acc{ext}"
    save_single_csv(acc_filename, times, final_accelerations, header)

def record_mode(filename, topic, compare_mode=False):
    # Import rospy here to allow 'plot' and 'test' modes without ROS
    try:
        import rospy
        from sensor_msgs.msg import JointState
    except ImportError:
        print("Error: rospy or sensor_msgs not found. Cannot record.")
        sys.exit(1)

    try:
        rospy.init_node('joint_monitor_recorder', anonymous=True)
    except rospy.exceptions.ROSException:
        pass # Node might already be initialized if running in some environments
    
    if compare_mode:
        # Subscribe to both master and puppet topics
        master_topic = topic.replace('/puppet/', '/master/')
        print(f"Compare Mode: Listening on puppet topic {topic} and master topic {master_topic}...")
        print(f"Press Ctrl+C to stop recording")
        print(f"Will generate 4 files:")
        root, ext = os.path.splitext(filename)
        print(f"  1. {filename} - Puppet position")
        print(f"  2. {root}_vel{ext} - Puppet velocity")
        print(f"  3. {root}_acc{ext} - Puppet acceleration")
        print(f"  4. {root}_master{ext} - Master command (position only)")
        
        rospy.Subscriber(topic, JointState, callback)
        rospy.Subscriber(master_topic, JointState, master_callback)
        
        # Save on shutdown
        def on_shutdown():
            # Save puppet data (position, velocity, acceleration)
            save_csv(filename)
            # Save master data (position only, no velocity/acceleration)
            root, ext = os.path.splitext(filename)
            master_filename = f"{root}_master{ext}"
            save_position_only_csv(master_filename, master_data_records)
            
        rospy.on_shutdown(on_shutdown)
    else:
        print(f"Listening on {topic}...")
        print(f"Press Ctrl+C to stop recording and save to '{filename}'")
        
        rospy.Subscriber(topic, JointState, callback)
        
        # Save on shutdown
        def on_shutdown():
            save_csv(filename)
            
        rospy.on_shutdown(on_shutdown)
    
    rospy.spin()

def plot_mode(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    print(f"Reading from {filename}...")
    try:
        data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # skip header
            for row in reader:
                data.append([float(x) for x in row])
        
        data_np = np.array(data)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if data_np.size == 0:
        print("Empty data.")
        return

    time_axis = data_np[:, 0]
    joints_data = data_np[:, 1:]
    
    num_vals = joints_data.shape[1]
    # Identify number of joints to plot
    # Typically 7 for this robot (6 arm + 1 gripper)
    
    print(f"Plotting {num_vals} traces over {time_axis[-1]:.2f} seconds...")

    fig, axes = plt.subplots(num_vals, 1, sharex=True, figsize=(10, 2.5 * num_vals))
    if num_vals == 1:
        axes = [axes]
    
    for i in range(num_vals):
        ax = axes[i]
        
        # Naming convention: first 6 are arm joints, 7th is gripper
        label_name = f"joint{i}"

        ax.plot(time_axis, joints_data[:, i], label=label_name)
        ax.set_ylabel(label_name)
        ax.grid(True)
        ax.legend(loc='upper right')
        
        # Set Y limits for arm joints to [-pi, pi] only for position data
        if i < 6 and not ("_vel" in filename or "_acc" in filename):
            ax.set_ylim(-np.pi, np.pi)
            # Reference lines
            ax.axhline(y=np.pi, color='r', linestyle='--', alpha=0.3)
            ax.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.3)
            
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.1)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Joint States from {filename}")
    plt.tight_layout()
    
    plot_filename = filename.replace('.csv', '.png')
    if plot_filename == filename:
        plot_filename += ".png"
        
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

def compare_plot_mode(puppet_file, master_file):
    """Plot both master and puppet joint positions on the same figure"""
    if not os.path.exists(puppet_file):
        print(f"Puppet file not found: {puppet_file}")
        return
    if not os.path.exists(master_file):
        print(f"Master file not found: {master_file}")
        return

    print(f"Reading puppet data from {puppet_file}...")
    print(f"Reading master data from {master_file}...")
    
    try:
        # Read puppet data
        puppet_data = []
        with open(puppet_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                puppet_data.append([float(x) for x in row])
        puppet_np = np.array(puppet_data)
        
        # Read master data
        master_data = []
        with open(master_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                master_data.append([float(x) for x in row])
        master_np = np.array(master_data)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if puppet_np.size == 0 or master_np.size == 0:
        print("Empty data.")
        return

    puppet_time = puppet_np[:, 0]
    puppet_joints = puppet_np[:, 1:]
    
    master_time = master_np[:, 0]
    master_joints = master_np[:, 1:]
    
    num_joints = puppet_joints.shape[1]
    
    print(f"Comparing {num_joints} joints...")

    fig, axes = plt.subplots(num_joints, 1, sharex=True, figsize=(12, 2.5 * num_joints))
    if num_joints == 1:
        axes = [axes]
    
    for i in range(num_joints):
        ax = axes[i]
        
        # Plot both master (command) and puppet (actual)
        ax.plot(master_time, master_joints[:, i], 'b-', label=f'Master (Command)', alpha=0.7, linewidth=1.5)
        ax.plot(puppet_time, puppet_joints[:, i], 'r--', label=f'Puppet (Actual)', alpha=0.7, linewidth=1.5)
        
        ax.set_ylabel(f'Joint {i}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Set Y limits for arm joints to [-pi, pi]
        if i < 6:
            ax.set_ylim(-np.pi, np.pi)
            ax.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.2)
            ax.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.2)
            
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.1)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Master Command vs Puppet Actual Position")
    plt.tight_layout()
    
    # Generate output filename
    root, ext = os.path.splitext(puppet_file)
    compare_filename = f"{root}_compare.png"
        
    plt.savefig(compare_filename, dpi=150)
    print(f"Comparison plot saved to {compare_filename}")

def generate_test_data(filename, duration=60, step=0.02):
    print(f"Generating test data to {filename} (Duration: {duration}s)...")
    t = 0
    data = []
    
    # Simulating 7 joints: 6 arm joints + 1 gripper
    while t < duration:
        # 6 Arm Joints: Generate sine waves with different frequencies/phases
        pos_row = []
        vel_row = []
        
        for i in range(6):
            # Pos: A * sin(wt + phi)
            amp = 1.0 + 0.3 * (i % 3)
            freq = 0.2 + 0.1 * i
            phase = 0.5 * i
            val = amp * math.sin(2 * math.pi * freq * t + phase)
            
            # Vel: derivative -> A * w * cos(wt + phi)
            w = 2 * math.pi * freq
            vel = amp * w * math.cos(w * t + phase)
            
            pos_row.append(val)
            vel_row.append(vel)
        
        # Gripper
        gripper_val = 0.04 * (math.sin(0.5 * t) + 1)
        gripper_vel = 0.04 * 0.5 * math.cos(0.5 * t)
        
        pos_row.append(gripper_val)
        vel_row.append(gripper_vel)
        
        rec = {
            'time': t,
            'pos': pos_row,
            'vel': vel_row
        }
        data.append(rec)
        t += step
        
    global data_records
    data_records = data
    save_csv(filename)
    data_records = []
    print("Test data generated (including velocity).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor, record, and plot robot joint states.")
    parser.add_argument('mode', choices=['record', 'plot', 'test', 'compare'], 
                        help="Mode: record (ROS->CSV), plot (CSV->PNG), test (Gen Dummy CSV), compare (Plot master vs puppet)")
    
    # 获取脚本所在目录下的 datas 文件夹作为默认路径
    default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datas")
    parser.add_argument('--file', '-f', default=os.path.join(default_data_dir, "joint_data.csv"), help="CSV filename to read/write")
    parser.add_argument('--topic', '-t', default="/puppet/joint_left", help="ROS topic to subscribe to (record mode only). Use /puppet/joint_* for velocity data, /master/joint_* for position only.")
    parser.add_argument('--compare-mode', '-c', action='store_true', help="Record both master and puppet data for comparison")
    parser.add_argument('--master-file', '-m', help="Master CSV file for compare plot mode")
    
    args = parser.parse_args()
    
    if args.mode == 'record':
        record_mode(args.file, args.topic, args.compare_mode)
    elif args.mode == 'plot':
        plot_mode(args.file)
    elif args.mode == 'test':
        generate_test_data(args.file)
    elif args.mode == 'compare':
        # Determine master file
        if args.master_file:
            master_file = args.master_file
        else:
            root, ext = os.path.splitext(args.file)
            master_file = f"{root}_master{ext}"
        compare_plot_mode(args.file, master_file)



# 使用示例：
# # 生成3个文件：my_experiment.csv (position), my_experiment_vel.csv, my_experiment_acc.csv
# 
# # 2. 对比录制模式（同时录制 master 指令和 puppet 实际位置）
# python monitor_joints.py record -f left_arm_data.csv --compare-mode
# # 生成4个文件：
# #   - left_arm_data.csv (puppet绝对位置)
# #   - left_arm_data_vel.csv (puppet速度)
# #   - left_arm_data_acc.csv (puppet加速度)
# #   - left_arm_data_master.csv (master动作指令，仅position)
# 
# # 3. 绘制单个数据图表
# python monitor_joints.py plot -f left_arm_data.csv
# python monitor_joints.py plot -f left_arm_data_vel.csv
# python monitor_joints.py plot -f left_arm_data_acc.csv
# python monitor_joints.py plot -f left_arm_data_master
# python monitor_joints.py plot -f left_arm_data.csv
# python monitor_joints.py plot -f left_arm_data_vel.csv
# python monitor_joints.py plot -f left_arm_data_acc.csv
# 
# # 4. 绘制对比图（master指令 vs puppet实际位置）
# python monitor_joints.py compare -f left_arm_data.csv
# # 或者指定master文件
# python monitor_joints.py compare -f left_arm_data.csv -m left_arm_data_master.csv
# 
# # 5. 生成测试数据
# python monitor_joints.py test -f test_data.csv