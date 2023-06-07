#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rosbag

from geometry_msgs.msg import Point
from qt_nuitrack_app.msg import Skeletons

from mas_qt.tf_utils import TFUtils
from mas_qt_ros.skeleton_utils import JointUtils

class Skeleton(object):
    def __init__(self, id):
        self.id = id
        self.upper_body_points = []
        self.left_hand_points = []
        self.right_hand_points = []
        self.left_leg_points = []
        self.right_leg_points = []

    def clear_points(self):
        self.upper_body_points = []
        self.left_hand_points = []
        self.right_hand_points = []
        self.left_leg_points = []
        self.right_leg_points = []

    def draw_skeleton(self, ax):
        for i in range(0, len(self.upper_body_points)-1):
            ax.plot(*zip(self.upper_body_points[i], self.upper_body_points[i+1]), color='blue')

        for i in range(0, len(self.left_hand_points)-1):
            ax.plot(*zip(self.left_hand_points[i], self.left_hand_points[i+1]), color='blue')

        for i in range(0, len(self.right_hand_points)-1):
            ax.plot(*zip(self.right_hand_points[i], self.right_hand_points[i+1]), color='blue')

        for i in range(0, len(self.left_leg_points)-1):
            ax.plot(*zip(self.left_leg_points[i], self.left_leg_points[i+1]), color='blue')

        for i in range(0, len(self.right_leg_points)-1):
            ax.plot(*zip(self.right_leg_points[i], self.right_leg_points[i+1]), color='blue')

class SkeletonMatplotlibVisualiser(object):
    def __init__(self, figure_axes):
        self.figure_axes = figure_axes
        self.cam_base_link_translation = [0., 0., 0.]
        self.cam_base_link_rot = [np.pi/2, 0., np.pi/2]
        self.cam_base_link_tf = TFUtils.get_homogeneous_transform(self.cam_base_link_rot,
                                                                  self.cam_base_link_translation)
        self.skeleton_points = {}

    def get_skeleton_points(self, skeleton_collection_msg):
        for skeleton_msg in skeleton_collection_msg.skeletons:
            if skeleton_msg.id in self.skeleton_points:
                self.skeleton_points[skeleton_msg.id].clear_points()
            else:
                self.skeleton_points[skeleton_msg.id] = Skeleton(skeleton_msg.id)

            for joint in skeleton_msg.joints:
                # the joint msg contains the position in mm
                position = np.array(joint.real) / 1000.
                position_hom = np.array([[position[0]], [position[1]], [position[2]], [1.]])
                position_base_link = self.cam_base_link_tf.dot(position_hom)
                position_base_link = position_base_link.flatten()[0:3]

                position_base_link[1] = -position_base_link[1]

                joint_name = JointUtils.JOINTS[joint.type]
                if joint_name in JointUtils.JOINTS_TO_IGNORE:
                    continue

                if joint_name in JointUtils.BODY_JOINT_NAMES:
                    self.skeleton_points[skeleton_msg.id].upper_body_points.append((position_base_link[0],
                                                                                    position_base_link[1],
                                                                                    position_base_link[2]))
                elif joint_name in JointUtils.LEFT_ARM_JOINT_NAMES:
                    self.skeleton_points[skeleton_msg.id].left_hand_points.append((position_base_link[0],
                                                                                   position_base_link[1],
                                                                                   position_base_link[2]))
                elif joint_name in JointUtils.RIGHT_ARM_JOINT_NAMES:
                    self.skeleton_points[skeleton_msg.id].right_hand_points.append((position_base_link[0],
                                                                                    position_base_link[1],
                                                                                    position_base_link[2]))
                elif joint_name in JointUtils.LEFT_LEG_JOINT_NAMES:
                    self.skeleton_points[skeleton_msg.id].left_leg_points.append((position_base_link[0],
                                                                                  position_base_link[1],
                                                                                  position_base_link[2]))
                elif joint_name in JointUtils.RIGHT_LEG_JOINT_NAMES:
                    self.skeleton_points[skeleton_msg.id].right_leg_points.append((position_base_link[0],
                                                                                   position_base_link[1],
                                                                                   position_base_link[2]))


    def plot_skeletons(self):
        for skeleton_id in self.skeleton_points:
            self.skeleton_points[skeleton_id].draw_skeleton(self.figure_axes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise skeleton data from a rosbag')
    parser.add_argument('-i', '--participant-id', type=str, required=True,
                        help='ID of the participant')
    parser.add_argument('-b', '--bag-file', type=str, required=True,
                        help='Path to a rosbag file to read')
    parser.add_argument('-t', '--skeleton-topic-name', type=str,
                        default='/qt_nuitrack_app/skeletons',
                        help='Name of a logged skeleton topic')

    args = parser.parse_args()
    participant_id = args.participant_id
    bag_file_path = args.bag_file
    skeleton_topic_name = args.skeleton_topic_name

    figure = plt.figure(1)
    figure_axes = figure.add_subplot(111, projection='3d')
    figure_axes.view_init(elev=30, azim=135)

    skeleton_visualiser = SkeletonMatplotlibVisualiser(figure_axes)
    if not os.path.isdir(participant_id):
        os.mkdir(participant_id)

    print(f'Processing {bag_file_path}...')
    bag = rosbag.Bag(bag_file_path)
    msg_counter = 0
    try:
        for _, skeleton_collection_msg, _ in bag.read_messages(topics=[skeleton_topic_name]):
            figure_axes.clear()
            figure_axes.set_xlabel('x')
            figure_axes.set_ylabel('y')
            figure_axes.set_zlabel('z')
            figure_axes.set_xlim([-2,2])
            figure_axes.set_ylim([-2,2])
            figure_axes.set_zlim([-2,2])

            skeleton_visualiser.get_skeleton_points(skeleton_collection_msg)
            skeleton_visualiser.plot_skeletons()
            msg_counter += 1
            plt.savefig(f'{participant_id}/{msg_counter}.jpg')

            if (msg_counter % 1000) == 0:
                print(f'Processed {msg_counter} messages so far')
    except (KeyboardInterrupt, SystemExit):
        print('skeleton_visualiser interrupted; exiting...')