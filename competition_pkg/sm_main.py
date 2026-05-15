#!/usr/bin/env python3
# -*-encoding:UTF-8-*-

"""
File: sm_main.py
"""

# モジュールのインポート(ROS2関連)
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist

# モジュールのインポート（YASMIN関連）
# https://github.com/uleroboticsgroup/yasmin.git
from yasmin import StateMachine
from yasmin_viewer import YasminViewerPub

# モジュールのインポート（自作:各ステート）
from .state_main import patrol, report, finish


class StateMachineNode(Node):
    """StateMachineNodeクラス（Nodeクラスを継承）
    ステートマシンを実行するノードクラス
    """

    def __init__(self):
        """クラスの初期化メソッド"""
        super().__init__("sm_main")

        self.get_logger().info("\033[43m\033[30m\033[1m<< PLEASE ENTER TO START >>\033[0m")
        input()
        self.get_logger().info("Task Start!!")

        self.vel_pub = self.create_publisher(msg_type=Twist, topic="cmd_vel", qos_profile=10)

        # StateMachineクラスのインスタンスを生成（最終結果として "EXIT" を返す）
        sm = StateMachine(outcomes=["EXIT"])

        # 1. ステートマシンにPatrolStateを追加
        sm.add_state(
            name="Patrol",
            state=patrol.PatrolState(node=self),
            transitions={
                "outcome1": "Report",   # 赤色検知時はReportへ
                "outcome3": "Finish"    # 5周完了時はFinishへ
            },
        )

        # 2. ステートマシンにReportStateを追加
        sm.add_state(
            name="Report",
            state=report.ReportState(node=self),
            transitions={
                "outcome2": "Patrol"    # 物体除去・復帰後はPatrolへ戻る
            },
        )

        # 3. ステートマシンにFinishStateを追加
        sm.add_state(
            name="Finish",
            state=finish.FinishState(node=self),
            transitions={
                "succeed": "EXIT"       # 終了処理完了後はステートマシンを抜ける
            },
        )

        # Yasmin Viewerにステートマシンの情報をパブリッシュ
        YasminViewerPub(fsm_name="SM_MAIN", fsm=sm)

        # ステートマシンを実行
        outcome = sm()
        self.get_logger().info("State Machine finished with outcome: " + outcome)


def shutdown(node: Node):
    """シャットダウン関数
    終了時にTurtleBot3を停止させる
    """
    node.get_logger().info("State Machine Cleanup!!")
    pub = node.create_publisher(Twist, "cmd_vel", 10)
    pub.publish(Twist())
    node.get_clock().sleep_for(Duration(nanoseconds=100))
    node.destroy_publisher(pub)


def main(args=None):
    """Main関数"""
    rclpy.init(args=args)

    try:
        node = StateMachineNode()
    except KeyboardInterrupt:
        pass
    finally:
        # nodeが存在する場合のみシャットダウンを実行
        if 'node' in locals():
            shutdown(node)
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()