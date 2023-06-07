#!/usr/bin/env python3
from typing import Tuple, Sequence, List, Dict
import os
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

def get_joint_error_mean_and_std(target_joint_values: Dict[str, Sequence[float]],
                                 recorded_joint_values: Dict[str, Sequence[float]]) -> Tuple[List[float], List[float]]:
    joint_name = list(target_joint_values.keys())[0]
    target_value_count = len(target_joint_values[joint_name])

    # print(target_joint_values[joint_name])
    # print(recorded_joint_values[joint_name])

    joint_error_means = []
    joint_error_stds = []
    try:
        for i in range(1, target_value_count):
            # HACK: adding an offset of about two months, one day, an one hour and a half
            # new_target_t = target_joint_values[joint_name][i][0] - 5188525
            new_target_t = target_joint_values[joint_name][i][0]
            recordings_before_new_cmd = get_measurements_before_cmd(recorded_joint_values,
                                                                    new_target_t)
            previous_target_values = {joint:target_joint_values[joint][i-1][1]
                                      for joint in target_joint_values}
            errors = [previous_target_values[joint] - recordings_before_new_cmd[joint]
                      for joint in previous_target_values]
            joint_error_means.append(np.mean(errors))
            joint_error_stds.append(np.std(errors))
    except:
        print('Done')
    return np.array(joint_error_means), np.array(joint_error_stds)

def get_measurements_before_cmd(recorded_joint_values: Dict[str, Sequence[float]],
                                new_target_t: float) -> Dict[str, float]:
    """Returns the joint measurements at the last timestamp before the timestamp
    at which a new command has been sent to the robot.

    Keyword arguments:
    recorded_joint_values -- joint measurements as a dictionary of joint names and,
                             for each joint, a list of [timestamp, measurement] values
    new_target_t -- timestamp of a command

    """
    # we just take one of the joints for searching for the timestamp
    joint_name = list(recorded_joint_values.keys())[0]

    # we search for the timestamp
    value_idx = 0
    while recorded_joint_values[joint_name][value_idx][0] < new_target_t:
        # print(value_idx, new_target_t, recorded_joint_values[joint_name][value_idx][0])
        value_idx += 1

    # we extract the joint measurements at the timestamp of the recording
    measurements = {joint:recorded_joint_values[joint][value_idx-1][1]
                    for joint in recorded_joint_values}
    return measurements

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Directory with data files (as joblib dumps)')

    args = parser.parse_args()
    data_dir = args.data_dir

    for f_name in os.listdir(data_dir):
        f_path = os.path.join(data_dir, f_name)
        if os.path.isfile(f_path):
            action_name = os.path.splitext(f_name)[0]
            with open(f_path, 'rb') as f:
                target_values, recorded_values = joblib.load(f)
                action_error_means, action_error_stds = get_joint_error_mean_and_std(target_values, recorded_values)
                plt.plot(action_error_means, label=action_name)
                plt.fill(np.concatenate([range(0,len(action_error_means)), range(0,len(action_error_means))[::-1]]),
                         np.concatenate([action_error_means - action_error_stds,
                                        (action_error_means + action_error_stds)[::-1]]),
                         alpha=.2, ec='None')

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Frame', fontsize=30)
    plt.ylabel('Error [Deg]', fontsize=30)
    plt.xlim([0, 140])
    plt.grid()
    # plt.legend(fontsize=22, loc='upper right')
    plt.show()
