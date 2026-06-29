import rospy
from sensor_msgs.msg import JointState
import time
import sys

# 尝试导入 constants，如果失败则使用默认值
try:
    from agilex_cobot_magic import constants
except ImportError:
    try:
        import constants
    except ImportError:
        print("Warning: Could not import constants.py")

def get_gripper_state(msg):
    # 假设夹爪是最后一个关节
    return msg.position[-1], msg.name

def main():
    rospy.init_node('gripper_tester', anonymous=True)
    
    print("Waiting for joint states...")
    # 订阅左臂状态（假设左臂话题是 /puppet/joint_left，根据 robot_utils.py 推断）
    # 如果你的话题名不一样，请修改这里
    try:
        msg = rospy.wait_for_message("/puppet/joint_left", JointState, timeout=5.0)
    except rospy.ROSException:
        print("Timeout: Could not receive message from /puppet/joint_left")
        return

    current_pos, joint_names = get_gripper_state(msg)
    print(f"\n=== Current Gripper Status ===")
    print(f"All Joint Names: {joint_names}")
    print(f"Current Gripper Position (Last Joint): {current_pos}")
    print("==============================\n")

    # 创建发布者
    pub = rospy.Publisher('/master/joint_left', JointState, queue_size=10)
    time.sleep(1) # 等待连接

    print("Starting Range Test...")
    print("WARNING: The gripper will move! Please keep your hands away.")
    
    # 测试序列：从当前位置开始，尝试几个典型值
    # 根据之前的线索：复位是 3.55，constants里是 1.49
    test_values = [current_pos, 0.0, 1.0, 2.0, 3.0, 3.55, 4.0, -1.0]
    
    for val in test_values:
        print(f"Sending command: {val} ...")
        
        # 构造消息
        cmd_msg = JointState()
        cmd_msg.header.stamp = rospy.Time.now()
        # 使用从消息中获取的真实名字
        cmd_msg.name = msg.name 
        # 保持其他关节不动，只改夹爪
        cmd_pos = list(msg.position)
        cmd_pos[-1] = val
        cmd_msg.position = cmd_pos
        
        # 发布多次以确保收到
        for _ in range(10):
            pub.publish(cmd_msg)
            time.sleep(0.05)
            
        # 等待并读取新位置
        time.sleep(2.0)
        new_msg = rospy.wait_for_message("/puppet/joint_left", JointState)
        new_pos, _ = get_gripper_state(new_msg)
        print(f" -> Result position: {new_pos}")
        
        user_input = input("Press Enter to continue, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    main()
