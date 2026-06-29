#!/usr/bin/env python3
"""
相机录制工具 - 订阅ROS相机话题，录制推理过程中的视频

功能:
- 支持三视角相机可选录制 (front, left, right)
- Ctrl+C 结束录制
- 自动拼接多视角图像，主视角(front)放中间
- 保存为 MP4 视频文件

使用示例:
    # 录制全部三个相机（默认）
    python monitor_camera.py record -o inference_video.mp4
    
    # 只录制主视角
    python monitor_camera.py record --cameras front -o front_only.mp4
    
    # 录制双视角
    python monitor_camera.py record --cameras front left -o two_cameras.mp4
    
    # 指定帧率
    python monitor_camera.py record --fps 30 -o output.mp4
"""

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np

# =============================================================================
# 默认配置
# =============================================================================
DEFAULT_CAMERA_TOPICS = {
    'front': '/camera_f/color/image_raw',  # 前置相机（主视角）
    'left': '/camera_l/color/image_raw',   # 左腕相机
    'right': '/camera_r/color/image_raw',  # 右腕相机
}

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
DEFAULT_FPS = 30
DEFAULT_CAMERAS = ['front', 'left', 'right']

# =============================================================================
# 全局变量
# =============================================================================
frame_buffers = {}  # {camera_name: [(timestamp, frame), ...]}
buffer_locks = {}   # {camera_name: threading.Lock}
recording = True
start_time = None
bridge = None


def get_timestamp_from_msg(msg):
    """从 ROS 消息中获取时间戳"""
    ts = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
    if ts == 0:
        ts = time.time()
    return ts


class CameraRecorder:
    """相机录制器类"""
    
    def __init__(self, cameras, topics, output_file, fps, target_height=480):
        """
        初始化相机录制器
        
        Args:
            cameras: 要录制的相机列表 ['front', 'left', 'right']
            topics: 相机话题字典 {camera_name: topic}
            output_file: 输出视频文件路径
            fps: 视频帧率
            target_height: 目标图像高度（用于统一分辨率）
        """
        self.cameras = cameras
        self.topics = topics
        self.output_file = output_file
        self.fps = fps
        self.target_height = target_height
        
        self.frame_buffers = {cam: [] for cam in cameras}
        self.buffer_locks = {cam: threading.Lock() for cam in cameras}
        self.latest_frames = {cam: None for cam in cameras}
        
        self.recording = True
        self.start_time = None
        self.frame_count = 0
        
        self.bridge = None
        self.subscribers = []
        
    def init_ros(self):
        """初始化 ROS 节点和订阅器"""
        try:
            import rospy
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
        except ImportError as e:
            print(f"错误: 无法导入 ROS 相关模块: {e}")
            print("请确保已 source ROS 环境: source /opt/ros/noetic/setup.bash")
            sys.exit(1)
        
        self.bridge = CvBridge()
        
        try:
            rospy.init_node('camera_recorder', anonymous=True)
        except rospy.exceptions.ROSException:
            pass  # 节点可能已初始化
        
        # 为每个相机创建订阅器
        for cam_name in self.cameras:
            topic = self.topics.get(cam_name)
            if topic is None:
                print(f"警告: 未找到相机 '{cam_name}' 的话题配置")
                continue
                
            # 使用闭包捕获 cam_name
            def make_callback(name):
                def callback(msg):
                    self._image_callback(name, msg)
                return callback
            
            sub = rospy.Subscriber(topic, Image, make_callback(cam_name), queue_size=1)
            self.subscribers.append(sub)
            print(f"[订阅] {cam_name}: {topic}")
        
        print(f"\n开始录制... 按 Ctrl+C 结束\n")
        
    def _image_callback(self, cam_name, msg):
        """图像回调函数"""
        if not self.recording:
            return
            
        try:
            # 转换图像格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 获取时间戳
            timestamp = get_timestamp_from_msg(msg)
            
            if self.start_time is None:
                self.start_time = timestamp
            
            rel_time = timestamp - self.start_time
            
            # 存储帧
            with self.buffer_locks[cam_name]:
                self.frame_buffers[cam_name].append((rel_time, cv_image.copy()))
                self.latest_frames[cam_name] = cv_image.copy()
            
            # 定期打印状态
            total_frames = sum(len(buf) for buf in self.frame_buffers.values())
            if total_frames % 100 == 0:
                self._print_status()
                
        except Exception as e:
            print(f"[错误] 处理 {cam_name} 图像时出错: {e}")
    
    def _print_status(self):
        """打印录制状态"""
        status_parts = []
        for cam in self.cameras:
            with self.buffer_locks[cam]:
                count = len(self.frame_buffers[cam])
            status_parts.append(f"{cam}:{count}")
        
        elapsed = time.time() - (self.start_time if self.start_time else time.time())
        print(f"\r[录制中] 时长: {elapsed:.1f}s | 帧数: {' | '.join(status_parts)}", end="", flush=True)
    
    def stop_recording(self):
        """停止录制"""
        self.recording = False
        print(f"\n\n停止录制，正在处理视频...")
        
    def resize_to_height(self, img, target_height):
        """将图像缩放到指定高度，保持宽高比"""
        h, w = img.shape[:2]
        if h == target_height:
            return img
        scale = target_height / h
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    
    def create_placeholder(self, width, height):
        """创建占位图像（黑色）"""
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        # 添加 "No Signal" 文字
        cv2.putText(placeholder, "No Signal", (width//4, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        return placeholder
    
    def stitch_frames(self, frames_dict):
        """
        拼接多视角图像
        布局: [left] [front] [right] - 主视角在中间
        
        Args:
            frames_dict: {camera_name: frame} 字典
            
        Returns:
            拼接后的图像
        """
        # 确定目标高度
        target_h = self.target_height
        
        # 获取各相机图像，缩放到统一高度
        images = {}
        widths = {}
        
        for cam in ['left', 'front', 'right']:
            if cam in frames_dict and frames_dict[cam] is not None:
                img = self.resize_to_height(frames_dict[cam], target_h)
                images[cam] = img
                widths[cam] = img.shape[1]
            elif cam in self.cameras:
                # 该相机被选中但没有数据，使用占位符
                # 估计宽度（假设 4:3 或使用默认）
                est_width = int(target_h * 640 / 480)
                images[cam] = self.create_placeholder(est_width, target_h)
                widths[cam] = est_width
        
        # 按顺序拼接：left - front - right
        to_stitch = []
        order = ['left', 'front', 'right']
        
        for cam in order:
            if cam in images:
                to_stitch.append(images[cam])
        
        if not to_stitch:
            return None
            
        if len(to_stitch) == 1:
            return to_stitch[0]
        
        return np.hstack(to_stitch)
    
    def synchronize_and_stitch(self):
        """
        同步各相机的帧并拼接
        
        Returns:
            拼接后的帧列表 [(timestamp, stitched_frame), ...]
        """
        print("正在同步帧...")
        
        # 获取所有帧数据
        all_frames = {}
        for cam in self.cameras:
            with self.buffer_locks[cam]:
                all_frames[cam] = list(self.frame_buffers[cam])
        
        # 检查是否有数据
        total_frames = sum(len(frames) for frames in all_frames.values())
        if total_frames == 0:
            print("警告: 没有录制到任何帧")
            return []
        
        # 打印各相机帧数
        for cam, frames in all_frames.items():
            print(f"  {cam}: {len(frames)} 帧")
        
        # 确定主相机（优先使用 front，否则用第一个有数据的）
        primary_cam = None
        if 'front' in all_frames and len(all_frames['front']) > 0:
            primary_cam = 'front'
        else:
            for cam in self.cameras:
                if len(all_frames[cam]) > 0:
                    primary_cam = cam
                    break
        
        if primary_cam is None:
            return []
        
        print(f"使用 '{primary_cam}' 作为时间基准")
        
        # 基于主相机的时间戳同步
        stitched_frames = []
        primary_frames = all_frames[primary_cam]
        
        # 为每个相机构建时间索引
        time_indices = {}
        for cam in self.cameras:
            if cam != primary_cam and len(all_frames[cam]) > 0:
                time_indices[cam] = 0
        
        for ts, primary_frame in primary_frames:
            frames_dict = {primary_cam: primary_frame}
            
            # 为每个其他相机找最近的帧
            for cam in self.cameras:
                if cam == primary_cam:
                    continue
                if cam not in time_indices or len(all_frames[cam]) == 0:
                    continue
                
                cam_frames = all_frames[cam]
                idx = time_indices[cam]
                
                # 找到时间上最接近的帧
                while idx < len(cam_frames) - 1 and cam_frames[idx + 1][0] <= ts:
                    idx += 1
                
                # 检查前后哪个更近
                if idx < len(cam_frames):
                    if idx > 0:
                        diff_curr = abs(cam_frames[idx][0] - ts)
                        diff_prev = abs(cam_frames[idx - 1][0] - ts)
                        if diff_prev < diff_curr:
                            idx = idx - 1
                    
                    frames_dict[cam] = cam_frames[idx][1]
                    time_indices[cam] = idx
            
            # 拼接帧
            stitched = self.stitch_frames(frames_dict)
            if stitched is not None:
                stitched_frames.append((ts, stitched))
        
        print(f"同步完成，共 {len(stitched_frames)} 帧")
        return stitched_frames
    
    def save_video(self):
        """保存视频文件"""
        # 同步并拼接帧
        stitched_frames = self.synchronize_and_stitch()
        
        if not stitched_frames:
            print("错误: 没有可保存的帧")
            return False
        
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取视频尺寸
        first_frame = stitched_frames[0][1]
        height, width = first_frame.shape[:2]
        
        print(f"视频尺寸: {width}x{height}")
        print(f"帧率: {self.fps} fps")
        print(f"输出文件: {self.output_file}")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, self.fps, (width, height))
        
        if not out.isOpened():
            print("错误: 无法创建视频文件")
            return False
        
        # 写入帧
        print("正在写入视频...")
        for i, (ts, frame) in enumerate(stitched_frames):
            out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"\r  进度: {i + 1}/{len(stitched_frames)} ({100*(i+1)//len(stitched_frames)}%)", end="", flush=True)
        
        out.release()
        print(f"\n\n视频已保存: {self.output_file}")
        
        # 计算视频时长
        duration = len(stitched_frames) / self.fps
        print(f"视频时长: {duration:.2f} 秒")
        print(f"总帧数: {len(stitched_frames)}")
        
        # 保存元数据（可选）
        self._save_metadata(stitched_frames, width, height, duration)
        
        return True
    
    def _save_metadata(self, frames, width, height, duration):
        """保存录制元数据"""
        meta_file = self.output_file.replace('.mp4', '_info.txt')
        
        with open(meta_file, 'w') as f:
            f.write(f"录制时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"相机: {', '.join(self.cameras)}\n")
            f.write(f"分辨率: {width}x{height}\n")
            f.write(f"帧率: {self.fps} fps\n")
            f.write(f"总帧数: {len(frames)}\n")
            f.write(f"时长: {duration:.2f} 秒\n")
            f.write(f"\n话题:\n")
            for cam in self.cameras:
                f.write(f"  {cam}: {self.topics.get(cam, 'N/A')}\n")
        
        print(f"元数据已保存: {meta_file}")
    
    def run(self):
        """运行录制器"""
        import rospy
        
        self.init_ros()
        
        # 设置信号处理
        def signal_handler(sig, frame):
            self.stop_recording()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 开始录制
        try:
            while self.recording and not rospy.is_shutdown():
                rospy.sleep(0.1)
        except rospy.ROSInterruptException:
            pass
        
        # 保存视频
        self.save_video()


def record_mode(args):
    """录制模式"""
    # 解析相机列表
    cameras = args.cameras if args.cameras else DEFAULT_CAMERAS
    
    # 验证相机名称
    valid_cameras = []
    for cam in cameras:
        cam = cam.lower()
        if cam in DEFAULT_CAMERA_TOPICS:
            valid_cameras.append(cam)
        else:
            print(f"警告: 未知的相机 '{cam}'，已忽略")
    
    if not valid_cameras:
        print("错误: 没有有效的相机")
        sys.exit(1)
    
    # 构建话题字典
    topics = {}
    for cam in valid_cameras:
        if cam == 'front' and args.front_topic:
            topics[cam] = args.front_topic
        elif cam == 'left' and args.left_topic:
            topics[cam] = args.left_topic
        elif cam == 'right' and args.right_topic:
            topics[cam] = args.right_topic
        else:
            topics[cam] = DEFAULT_CAMERA_TOPICS[cam]
    
    # 确定输出文件
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"recording_{timestamp}.mp4")
    
    # 确保输出文件有 .mp4 后缀
    if not output_file.endswith('.mp4'):
        output_file += '.mp4'
    
    # 创建录制器并运行
    recorder = CameraRecorder(
        cameras=valid_cameras,
        topics=topics,
        output_file=output_file,
        fps=args.fps,
        target_height=args.height
    )
    
    recorder.run()


def list_topics_mode(args):
    """列出可用话题"""
    print("默认相机话题配置:")
    print("-" * 50)
    for cam, topic in DEFAULT_CAMERA_TOPICS.items():
        print(f"  {cam:8s}: {topic}")
    print("-" * 50)
    print("\n使用 'rostopic list | grep camera' 查看当前可用的相机话题")


def main():
    parser = argparse.ArgumentParser(
        description="相机录制工具 - 录制推理过程中的多视角视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 录制全部三个相机
  python monitor_camera.py record -o my_video.mp4
  
  # 只录制主视角
  python monitor_camera.py record --cameras front -o front_only.mp4
  
  # 录制前置和左腕相机
  python monitor_camera.py record --cameras front left -o two_cams.mp4
  
  # 指定帧率和分辨率
  python monitor_camera.py record --fps 25 --height 360 -o output.mp4
  
  # 查看默认话题配置
  python monitor_camera.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # record 子命令
    record_parser = subparsers.add_parser('record', help='录制视频')
    record_parser.add_argument('-o', '--output', type=str, default=None,
                               help='输出视频文件路径 (默认: videos/recording_时间戳.mp4)')
    record_parser.add_argument('--cameras', nargs='+', choices=['front', 'left', 'right'],
                               default=None, help='要录制的相机 (默认: 全部)')
    record_parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                               help=f'视频帧率 (默认: {DEFAULT_FPS})')
    record_parser.add_argument('--height', type=int, default=480,
                               help='目标图像高度，用于统一分辨率 (默认: 480)')
    record_parser.add_argument('--front-topic', type=str, default=None,
                               help='自定义前置相机话题')
    record_parser.add_argument('--left-topic', type=str, default=None,
                               help='自定义左腕相机话题')
    record_parser.add_argument('--right-topic', type=str, default=None,
                               help='自定义右腕相机话题')
    
    # list 子命令
    list_parser = subparsers.add_parser('list', help='列出默认话题配置')
    
    args = parser.parse_args()
    
    if args.mode == 'record':
        record_mode(args)
    elif args.mode == 'list':
        list_topics_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# =============================================================================
# 使用说明
# =============================================================================
#
# 1. 确保已 source ROS 环境:
#    source /opt/ros/noetic/setup.bash
#    source ~/agilex_ws/devel/setup.bash
#
# 2. 确保相机节点已启动
#
# 3. 运行录制:
#    python monitor_camera.py record -o inference_video.mp4
#
# 4. 按 Ctrl+C 结束录制，视频会自动保存
#
# =============================================================================
