#!/usr/bin/env python3
# -*-encoding:UTF-8-*-

"""
File: finish.py
"""

# モジュールのインポート(ROS2関連)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# モジュールのインポート（YASMIN関連）
from yasmin import State
from yasmin import Blackboard


class FinishState(State):
    """FinishStateクラス（Stateクラスの継承）"""

    def __init__(self, node: Node):
        """クラスの初期化メソッド"""
        super().__init__(outcomes=["succeed"])
        self.node = node
        self.vel_pub = self.node.create_publisher(msg_type=Twist, topic="cmd_vel", qos_profile=10)

    def execute(self, blackboard: Blackboard) -> str:
        """FINISHステートの実行メソッド"""
        self.node.get_logger().info("Executing state FINISH")

        # ロボットの停止コマンドを送信
        self.vel_pub.publish(Twist())
        self.node.get_logger().info("Stop!!")

        return "succeed"