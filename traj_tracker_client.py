import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
import modern_robotics as mr
from sensor_msgs.msg import JointState
import numpy as np
import csv
import sys


class TrajTrackerClient(Node):
    def __init__(self, csv_file, robot_namespace='/mobile_wx250s', topic_joint_states: str = 'mobile_wx250s/joint_states'):
        super().__init__('trajectory_client')
        self._action_client = ActionClient(self, FollowJointTrajectory, robot_namespace + '/arm_controller/follow_joint_trajectory')
        self.csv_file = csv_file
        self.robot_des: mrd.ModernRoboticsDescription = getattr(mrd, 'mobile_wx250s')
        self.topic_joint_states = topic_joint_states
        self.joint_states: JointState = None

        # Sottoscrivo al topic dei joint_states
        self.subscription = self.create_subscription(
            JointState,
            self.topic_joint_states,
            self.joint_state_callback,
            10
        )

    def joint_state_callback(self, msg: JointState):
        self.joint_states = msg

    def read_csv_trajectory(self):
        points = []
        with open(self.csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                point = JointTrajectoryPoint()
                # Imposta time_from_start dal valore "time" del file
                time_sec = float(row['time'])
                point.time_from_start.sec = int(time_sec)  # sec deve essere int
                point.time_from_start.nanosec = int((time_sec - int(time_sec)) * 1e9)  # nanosecondi
                # Estrae le posizioni per ciascun giunto
                positions = []
                for joint in ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']:
                    positions.append(float(row[joint]))
                point.positions = positions
                points.append(point)
        return points

    def send_goal(self):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
        goal_msg.trajectory.points = self.read_csv_trajectory()
        points = goal_msg.trajectory.points

        self.get_logger().info("Invio del goal della traiettoria...\n")
        print("Posizioni lette:\n")
        for point in points:
            t = point.time_from_start.sec
            print(f"t = {t}.{point.time_from_start.nanosec}, ")
            print(f"waist: {point.positions[0]:.3f}, shoulder: {point.positions[1]:.3f}, elbow: {point.positions[2]:.3f}, "
                  f"forearm_roll: {point.positions[3]:.3f}, wrist_angle: {point.positions[4]:.3f}, wrist_rotate: {point.positions[5]:.3f}")

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def get_ee_pose(self) -> np.ndarray:
        """
        Calcola la posa dell'end-effector rispetto al frame base (Space frame)
        :return: Matrice di trasformazione 4x4
        """
        js_index_map = dict(zip(self.joint_states.name, range(len(self.joint_states.name))))
        actual_joint_states = [
            self.joint_states.position[js_index_map[name]]
            for name in ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
        ]
        return mr.FKinSpace(self.robot_des.M, self.robot_des.Slist, actual_joint_states)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Il goal è stato rifiutato")
            return
        self.get_logger().info("Il goal è stato accettato")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("Trajectory tracking terminato con successo")
        
        # Attendo i joint_states e stampo la posa
        while rclpy.ok() and self.joint_states is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        T_ee = self.get_ee_pose()
        print("\nEnd-Effector Pose (matrice 4x4):")
        np.set_printoptions(precision=3, suppress=True)
        print(T_ee)
        
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().debug(f"Feedback: {feedback}")  # solo se vuoi vederli in log level debug


def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Uso: ros2 run traj_tracking_py traj_tracker_client.py path/to/trajectory.csv")
        return
    csv_file = sys.argv[1]
    trajectory_client = TrajTrackerClient(csv_file)
    trajectory_client.send_goal()
    rclpy.spin(trajectory_client)
    trajectory_client.destroy_node()


if __name__ == '__main__':
    main()
