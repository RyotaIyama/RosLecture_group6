#!/usr/bin/env python3
# -*-encoding:UTF-8-*-

"""
File: patrol.py
"""

# モジュールのインポート(外部)
import numpy as np
import cv2
import math

# モジュールのインポート(ROS2関連)
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped

# モジュールのインポート（YASMIN関連）
# https://github.com/uleroboticsgroup/yasmin.git
from yasmin import State
from yasmin import Blackboard


class PatrolState(State):
    """PatrolStateクラス（Stateクラスの継承）
    赤色領域を取得して追従する。
    4つの中継地点を巡回し、赤色物体を検知したらoutcome1に移行する。
    5周したらoutcom3に移行（終了）。
    """

    def __init__(self, node: Node):
        """クラスの初期化メソッド"""
        #  赤検出時は'outcome1'、5周終了時は 'outcome3' を返す
        super().__init__(outcomes=["outcome1", "outcome3"])
        self.node = node

        self.bridge = CvBridge()
        self.image_pub = self.node.create_publisher(
            msg_type=Image, topic="masked_image", qos_profile=10
        )
        self.image_sub = self.node.create_subscription(
            msg_type=Image, topic="image_raw", callback=self.callback, qos_profile=10
        )

        self.vel_pub = self.node.create_publisher(msg_type=Twist, topic="cmd_vel", qos_profile=10)
        self.goal_pub = self.node.create_publisher(msg_type=PoseStamped, topic="goal_pose", qos_profile=10)

        self.cmd_vel = Twist()
        self.detect_log = "stop"

# 赤色検知フラグ
        self.red_detected = False

        # --- 巡回ルートの設定（4つの中継地点） ---
        self.waypoints = []
        self.setup_waypoints()
        
        self.current_wp_idx = 0
        self.total_waypoints_visited = 0
        self.max_waypoints = len(self.waypoints) * 5

    def setup_waypoints(self):
        """RViz2の座標系に合わせた4つの中継地点を定義"""
        # 環境に合わせて座標(x, y)の数値を変更してください
        points = [
            (1.0, 0.0),  # 中継地点1
            (1.0, 1.0),  # 中継地点2
            (0.0, 1.0),  # 中継地点3
            (0.0, 0.0)   # 中継地点4
        ]
        for x, y in points:
            wp = PoseStamped()
            wp.header.frame_id = 'map'
            wp.pose.position.x = x
            wp.pose.position.y = y
            wp.pose.orientation.w = 1.0
            self.waypoints.append(wp)

    def callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.node.get_logger().info(e)

        # ========================= state =====================================
        # 赤色のマスク処理
        # =====================================================================
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, w, ch = hsv.shape
        hsv1 = hsv

        # 赤色の範囲1
        hsv_min = np.array([0, 150, 150])
        hsv_max = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

        # 赤色の範囲2
        hsv_min = np.array([160, 150, 150])
        hsv_max = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

        # 赤色領域のマスク
        mask = mask1 + mask2
        masked_hsv = cv2.bitwise_and(hsv1, hsv1, mask=mask)

        # 赤色領域の計算
        ones = np.ones((h, w))
        masked = cv2.bitwise_and(ones, ones, mask=mask)

        # 左、中、右の赤色領域を計算
        ones_left = sum(sum(masked[0:h, 0 : int(w / 3)]))
        ones_center = sum(sum(masked[0:h, int(w / 3) : int(2 * w / 3)]))
        ones_right = sum(sum(masked[0:h, int(2 * w / 3) : w]))

        # 赤色ピクセルの合計値を算出し、検知フラグを制御する
        total_red_area = ones_left + ones_center + ones_right
        
        # 画面内に合計500ピクセル以上の赤があれば検知とみなす
        if total_red_area > 500:
            self.red_detected = True  # フラグをTrueにする
            
            # cmd_vel の設定
            cmd_vel = Twist()
            if (ones_left > ones_center) and (ones_left > ones_right):
                detect_log = "Left side"
                cmd_vel.linear.x = 0.00
                cmd_vel.angular.z = 0.20
            elif (ones_center > ones_left) and (ones_center > ones_right):
                detect_log = "Center"
                cmd_vel.linear.x = 0.05
                cmd_vel.angular.z = 0.00
            elif (ones_right > ones_left) and (ones_right > ones_center):
                detect_log = "Right side"
                cmd_vel.linear.x = 0.00
                cmd_vel.angular.z = -0.20
            else:
                detect_log = "stop"
                cmd_vel.linear.x = 0.00
                cmd_vel.angular.z = 0.00
        else:
            self.red_detected = False
            detect_log = "stop"
            cmd_vel = Twist()

        self.detect_log = detect_log
        self.cmd_vel = cmd_vel

        # 結果のパブリッシュ
        try:
            img_cv = cv2.cvtColor(masked_hsv, cv2.COLOR_HSV2BGR)
            img_msg = self.bridge.cv2_to_imgmsg(img_cv, "bgr8")
            self.image_pub.publish(img_msg)
        except CvBridgeError as e:
            self.node.get_logger().info(e)

    def send_goal(self, pose_stamped: PoseStamped):
        """RViz2 / Nav2 へ目標中継地点を送信"""
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        self.goal_pub.publish(pose_stamped)
        self.node.get_logger().info(f"Send Goal: WP {self.current_wp_idx + 1}")

    def execute(self, blackboard: Blackboard) -> str:
        """
        PATROLステートの実行メソッド
        """
        self.node.get_logger().info("Patrol")
        self.node.get_logger().info("Start!!")
        
        # フラグの初期化
        self.red_detected = False
        
        # 最初の中継地点を送信
        self.send_goal(self.waypoints[self.current_wp_idx])

        while rclpy.ok():
            self.node.get_logger().info(self.detect_log)

            # 1. 赤い物体を検知したら即座に outcome1 へ移行
            if self.red_detected:
                self.node.get_logger().warn("Red object detected! Shifting to outcome1.")
                current_wp = self.waypoints[self.current_wp_idx]
                blackboard.target_x = current_wp.pose.position.x
                blackboard.target_y = current_wp.pose.position.y
                # ロボットを停止させる
                self.vel_pub.publish(Twist())
                return "outcome1"

            # 2. 5周（計20回中継地点を訪問）したら終了
            if self.total_waypoints_visited >= self.max_waypoints:
                break

            # 周期待機とスピン
            self.node.get_clock().sleep_for(Duration(seconds=1))
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
        # ロボットの停止コマンドを送信
        # 5周完了後の停止処理
        self.vel_pub.publish(Twist())
        self.node.get_logger().info("Patrol Completed (5 Loops done). Stop!!")
        self.node.get_clock().sleep_for(Duration(seconds=1))
        
        return "outcome3"