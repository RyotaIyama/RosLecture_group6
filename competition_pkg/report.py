#!/usr/bin/env python3
# -*-encoding:UTF-8-*-

"""
File: report.py
"""

# モジュールのインポート(外部)
import numpy as np
import cv2

# モジュールのインポート(ROS2関連)
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from geometry_msgs.msg import PoseStamped, Twist
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import Image
from turtlebot3_msgs.msg import Sound  # TurtleBot3のブザー用
from cv_bridge import CvBridge, CvBridgeError
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action._navigate_to_pose import (
    NavigateToPose_GetResult_Response,
    NavigateToPose_Feedback,
    NavigateToPose_FeedbackMessage,
)
import tf_transformations

# モジュールのインポート（YASMIN関連）
# https://github.com/uleroboticsgroup/yasmin.git
from yasmin import State
from yasmin import Blackboard


class ReportState(State):
    """ReportStateクラス（Stateクラスの継承）
    赤い物体検知時に音を出し、物体除去を待機する。
    除去後はNav2を用いて移動を再開し、完了したらoutcome2に移行する。
    """

    def __init__(self, node: Node):
        """クラスの初期化メソッド"""
        # 完了時はpatrolへ移行する 'outcome2' を返す
        super().__init__(outcomes=["outcome2"])

        # Nav2インスタンス変数の初期化
        self._goal_handle: ClientGoalHandle = None
        self._result_future: NavigateToPose_GetResult_Response = None
        self._feedback: NavigateToPose_Feedback = None
        self._status: int = None

        self.node = node
        self.bridge = CvBridge()
        
        # 状態管理・検知フラグ
        self.object_still_present = True

        # アクションクライアントの設定
        self.nav_to_pose_client = ActionClient(
            node=self.node, action_type=NavigateToPose, action_name="/navigate_to_pose"
        )

        # パブリッシャ / サブスクライバの設定
        self.sound_pub = self.node.create_publisher(msg_type=Sound, topic="sound", qos_profile=10)
        self.vel_pub = self.node.create_publisher(msg_type=Twist, topic="cmd_vel", qos_profile=10)
        self.image_sub = self.node.create_subscription(
            msg_type=Image, topic="image_raw", callback=self.image_callback, qos_profile=10
        )

    def image_callback(self, msg: Image):
        """赤い物体が画面から取り除かれたか監視するコールバック"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.node.get_logger().info(e)
            return

        # 赤色のマスク処理 (PatrolStateと同じHSVしきい値)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, w, ch = hsv.shape

        # 赤色の範囲1 & 2
        mask1 = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 150, 150]), np.array([179, 255, 255]))
        mask = mask1 + mask2

        # 赤色領域の計算
        ones = np.ones((h, w))
        masked = cv2.bitwise_and(ones, ones, mask=mask)
        total_red_area = sum(sum(masked))

        # 赤色ピクセルが著しく減少（例: 100ピクセル未満）したら除去されたとみなす
        if total_red_area < 100:
            self.object_still_present = False
        else:
            self.object_still_present = True

    def play_sound(self):
        """TurtleBot3のブザーを鳴らす（警告音）"""
        msg = Sound()
        msg.value = Sound.ERROR  # エラー警告音
        self.sound_pub.publish(msg)

    def goToPose(self, x: float, y: float, yaw: float) -> bool:
        """指定した目的地までナビゲーションするメソッド"""
        self.node.get_logger().debug("Waiting for 'NavigateToPose' action server")

        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().info("'NavigateToPose' action server not available, waiting...")

        pose = PoseStamped()
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quat[1]
        pose.pose.orientation.y = quat[2]
        pose.pose.orientation.z = quat[3]
        pose.pose.orientation.w = quat[0]

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.node.get_logger().info(f"Resuming navigation to goal: (x, y, yaw) = ({x}, {y}, {yaw})")

        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal=goal_msg,
            feedback_callback=self._feedbackcallback,
        )
        rclpy.spin_until_future_complete(self.node, send_goal_future)

        self._goal_handle = send_goal_future.result()

        if not self._goal_handle.accepted:
            self.node.get_logger().error(f"Goal to ({x}, {y}) was rejected!")
            return False

        self._result_future = self._goal_handle.get_result_async()
        return True

    def cancelNav(self):
        """実行中のナビゲーションをキャンセルするメソッド"""
        self.node.get_logger().info("Canceling current task.")
        if self._result_future:
            future = self._goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self.node, future)

    def isNavComplete(self) -> bool:
        """ナビゲーションの完了状態を確認するメソッド"""
        if not self._result_future:
            return True
        rclpy.spin_until_future_complete(self.node, self._result_future, timeout_sec=0.10)
        if self._result_future.result():
            self._status = self._result_future.result().status
            if self._status != GoalStatus.STATUS_SUCCEEDED:
                self.node.get_logger().debug(f"Task failed with status code: {self._status}")
                return True
        else:
            return False
        self.node.get_logger().debug("Navigation succeeded!")
        return True

    def getResult(self) -> int:
        return self._status

    def getFeedback(self) -> NavigateToPose_Feedback:
        return self._feedback

    def _feedbackcallback(self, msg: NavigateToPose_FeedbackMessage):
        self._feedback = msg.feedback

    def execute(self, blackboard: Blackboard) -> str:
        """
        REPORTステートの実行メソッド
        """
        self.node.get_logger().info("Executing state REPORT")
        
        # 1. 安全のためロボットを完全停止
        self.vel_pub.publish(Twist())

        # 2. 警告音を鳴らす
        self.play_sound()
        self.node.get_logger().warn("【警告】赤い物体を検知しました。除去を待機中...")

        # 3. 物体が取り除かれるまでその場でループ待機
        rate = self.node.create_rate(2)  # 0.5秒周期
        while rclpy.ok() and self.object_still_present:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            rate.sleep()

        self.node.get_logger().info("物体が取り除かれました。3秒後に移動を再開します。")
        self.node.get_clock().sleep_for(Duration(seconds=3))

        # 4. パトロールの次の中継地点へ向けてナビゲーションを再始動
        # ※本来はPatrolState側でblackboardに保存した「次に向かうべき目標座標」を設定します
        # ここでは例として(x=1.0, y=1.0, yaw=0.0)に移動する記述にしています
        target_x = getattr(blackboard, 'target_x', 1.0)
        target_y = getattr(blackboard, 'target_y', 1.0)
        
        self.goToPose(x=target_x, y=target_y, yaw=0.0)

        # 5. ナビゲーションが完了（あるいはタイムアウト）するまで同期待機
        while not self.isNavComplete():
            feedback = self.getFeedback()
            if feedback and feedback.navigation_time > 30:
                self.cancelNav()

        # ナビゲーション結果の判定
        result = self.getResult()
        if result == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Navigation succeeded! Returning to PatrolState.")
        else:
            self.node.get_logger().error("Navigation failed or was canceled.")

        # 次の状態（PatrolState）へ復帰するためにoutcome2を返す
        return "outcome2"