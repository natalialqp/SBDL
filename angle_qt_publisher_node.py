#!/usr/bin/env python3
import sys
from telnetlib import PRAGMA_HEARTBEAT
from time import sleep
import rospy
import pandas as pd
import joblib
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import csv
from data_filtering import Filters
from collition_avoidance import SelfcollitionAvoidanceModule
from data_processing import DataProcessing
import numpy as np  
import copy 

class QTImitationExecutor(object):

    def __init__(self, participant, action):
        '''
        This function is executed when the program starts
        The values of Kp, Ki and Kd were found experimentally
        Target_positions: stores the desired positions
        Calculated_angles: stores the angles corresponding to the skeleton
        Filtered_angles: stores all angles after the median filter
        Recorded_positions: stores the final angular positions after the controller
        Compensated_angles: stores the proposed angles after checking for collisions
        Joint_names: stores the names of all joints
        Instant_positions, robot_angles, desired_angle: are support dictionaries while using the controller
        '''
        self.target_positions = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
                                 'shoulder_left_pitch': [], 'shoulder_left_roll': [],
                                 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.calculated_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [], 'shoulder_left_pitch': [], 
                                  'shoulder_left_roll': [], 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.filtered_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [], 'shoulder_left_pitch': [], 
                                 'shoulder_left_roll': [], 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.recorded_positions = {'shoulder_right_pitch': [], 'shoulder_right_roll': [], 'shoulder_left_pitch': [],
                                   'shoulder_left_roll': [], 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.estimated_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [], 'shoulder_left_pitch': [], 
                                'shoulder_left_roll': [], 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.compensated_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [], 'shoulder_left_pitch': [],
                                 'shoulder_left_roll': [], 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.robot_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [], 'shoulder_left_pitch': [], 
                             'shoulder_left_roll': [], 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.instant_positions = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0, 'shoulder_left_pitch': 0,
                                  'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}             
        self.desired_angle = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0, 'shoulder_left_pitch': 0, 
                              'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        self.joint_names = ['shoulder_right_pitch', 'shoulder_right_roll', 'shoulder_left_pitch', 
                            'shoulder_left_roll', 'neck_pitch', 'elbow_right', 'elbow_left']    
        self.participant = participant
        self.action = action
        self.Ki = 0.2
        self.Kp = 1
        self.Kd = 0.06
        self.read_skeleton_joints()
        self.right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size = 1)
        self.left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size = 1)
        self.head_pub = rospy.Publisher('/qt_robot/head_position/command', Float64MultiArray, queue_size = 1)

    def read_angle_from_csv(self, name):
        '''
        This function reads the values of the joints to reproduce the movements
        Name: is the name of the file
        '''
        angle_vec = []
        with open(name, 'r') as file:                             
            reader = csv.reader(file)
            for row in reader:
                angle_vec.append(float(row[0])) 
        return angle_vec

    def addition(self, dict_1, dict_2):
        '''
        This function performs the sum of all the values of two dictionaries
        Input: dict_1, dict_2
        Output: dict_res
        '''
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict_1[i] + dict_2[i]
        return dict_res

    def division(self, dict, num):
        '''
        This function performs the division of all the values of a dictionary by a number
        Input: dict, num
        Output: dict_res
        '''
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict[i]/ num
        return dict_res

    def substraction(self, dict_1, dict_2):
        '''
        This function performs the substraction of all the values of a dictionary
        Input: dict_1, dict_2
        Output: dict_res
        '''
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict_1[i] - dict_2[i] 
        return dict_res

    def multiplication(self, dict, num):
        '''
        This function performs the multiplication of all the values of a dictionary for a number
        Input: dict
        Output: num
        '''
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict[i] * num 
        return dict_res

    def assignment(self, dict):
        '''
        This function saves  the estimated_angles in the controller
        Input: dict
        '''
        for i in self.joint_names:
            self.estimated_angles[i].append(dict[i])

    def qt_joint_state_cb(self, joint_state_msg):
        '''
        This function saves the recorded_positions taken from the robot's sensors
        Input: joint_state_msg
        instant_positions: saves the same information for the PID loop
        '''
        current_time = joint_state_msg.header.stamp.secs
        self.recorded_positions['neck_pitch'].append([current_time, joint_state_msg.position[0]])
        self.recorded_positions['elbow_left'].append([current_time, joint_state_msg.position[2]])
        self.recorded_positions['shoulder_left_pitch'].append([current_time, joint_state_msg.position[3]])
        self.recorded_positions['shoulder_left_roll'].append([current_time, joint_state_msg.position[4]])
        self.recorded_positions['elbow_right'].append([current_time, joint_state_msg.position[5]])
        self.recorded_positions['shoulder_right_pitch'].append([current_time, joint_state_msg.position[6]])
        self.recorded_positions['shoulder_right_roll'].append([current_time, joint_state_msg.position[7]])
        
        self.instant_positions['neck_pitch'] = joint_state_msg.position[0]
        self.instant_positions['elbow_left'] = joint_state_msg.position[2]
        self.instant_positions['shoulder_left_pitch'] = joint_state_msg.position[3]
        self.instant_positions['shoulder_left_roll'] = joint_state_msg.position[4]
        self.instant_positions['elbow_right'] = joint_state_msg.position[5]
        self.instant_positions['shoulder_right_pitch'] = joint_state_msg.position[6]
        self.instant_positions['shoulder_right_roll'] = joint_state_msg.position[7]

    def PID(self):
        '''
        This function uses the PID of the robot to estimate the angular positions
        '''
        # initialize stored data
        e_prev = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0, 'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        e = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0, 'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        e_integral = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0, 'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        current_angle = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0, 'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        delta_t = 0.01
        nothing = 0
        i = 0
        sequence_len = len(self.robot_angles['elbow_left'])
        while i < sequence_len:
            self.desired_angle['shoulder_right_roll']  = self.robot_angles['shoulder_right_roll'][i]
            self.desired_angle['shoulder_right_pitch'] = self.robot_angles['shoulder_right_pitch'][i]
            self.desired_angle['shoulder_left_roll']   = self.robot_angles['shoulder_left_roll'][i]
            self.desired_angle['shoulder_left_pitch']  = self.robot_angles['shoulder_left_pitch'][i]
            self.desired_angle['neck_pitch']           = self.robot_angles['neck_pitch'][i]
            self.desired_angle['elbow_right']          = self.robot_angles['elbow_right'][i]
            self.desired_angle['elbow_left']           = self.robot_angles['elbow_left'][i]
            self.save_target_data(i)

            #read angles from robot and save at self.instant_positions
            self.joint_state_sub = rospy.Subscriber('/qt_robot/joints/state', JointState, self.qt_joint_state_cb)
            while self.instant_positions['elbow_right'] == 0:
                nothing += 1
            current_angle = self.instant_positions.copy()  
            e = self.substraction(self.desired_angle, current_angle)
            self.assignment(current_angle)
            self.instant_positions['elbow_right'] = 0

            # PID calculations
            e_dot = self.division(self.substraction(e, e_prev), delta_t)
            e_integral = self.addition(e_integral, self.multiplication(e, delta_t))
            control_signal = self.addition(self.addition(self.multiplication(e, self.Kp), self.multiplication(e_dot, self.Kd)), self.multiplication(e_integral, self.Ki))
            current_angle = self.addition(current_angle, control_signal)
                
            # Publish angles to robot
            ref_right = Float64MultiArray()
            ref_left = Float64MultiArray()
            ref_head = Float64MultiArray()

            ref_right.data = [current_angle['shoulder_right_pitch'], current_angle['shoulder_right_roll'], current_angle['elbow_right']]
            ref_left.data = [current_angle['shoulder_left_pitch'], current_angle['shoulder_left_roll'], current_angle['elbow_left']]
            ref_head.data = [0, current_angle['neck_pitch']]
                
            imitation_executor.right_pub.publish(ref_right)
            imitation_executor.left_pub.publish(ref_left)
            imitation_executor.head_pub.publish(ref_head)

            # update stored data for next iteration
            e_prev = e
            i += 1
        self.save_graphs('Robot angles vs. PID signal ', self.robot_angles, self.estimated_angles, 'Robot calculated signal', 'Robot PID signal')

    def PID_online(self):
        '''
        This function uses the build PID to estimate the angular positions
        '''
        # initialize stored data
        nothing = 0
        self.joint_state_sub = rospy.Subscriber('/qt_robot/joints/state', JointState, self.qt_joint_state_cb)
        while self.instant_positions['elbow_right'] == 0:
            nothing += 1
        current_angle = self.instant_positions.copy()
        self.instant_positions['elbow_right'] = 0

        delta_t = 1
        i = 0
        epsilon = 1
        sequence_len = len(self.robot_angles['elbow_left'])
        ref_right = Float64MultiArray()
        ref_left = Float64MultiArray()
        ref_head = Float64MultiArray()
        ref_right.data = [-90, -60, -35]
        ref_left.data = [90, -60, -35]
        ref_head.data = [0, 1]
        imitation_executor.right_pub.publish(ref_right)
        imitation_executor.left_pub.publish(ref_left)
        imitation_executor.head_pub.publish(ref_head)
        rospy.sleep(3)

        while i < sequence_len:
            self.desired_angle['shoulder_right_roll']  = self.robot_angles['shoulder_right_roll'][i]
            self.desired_angle['shoulder_right_pitch'] = self.robot_angles['shoulder_right_pitch'][i]
            self.desired_angle['shoulder_left_roll']   = self.robot_angles['shoulder_left_roll'][i]
            self.desired_angle['shoulder_left_pitch']  = self.robot_angles['shoulder_left_pitch'][i]
            self.desired_angle['neck_pitch']           = self.robot_angles['neck_pitch'][i]
            self.desired_angle['elbow_right']          = self.robot_angles['elbow_right'][i]
            self.desired_angle['elbow_left']           = self.robot_angles['elbow_left'][i]
            self.save_target_data(i)

            for joint in self.joint_names:
                e_prev = 0
                e_integral = 0
                iter_pid = 0
                e = self.desired_angle[joint] - current_angle[joint]
                while abs(e) > epsilon and iter_pid < 20:
                    # PID calculations
                    e_dot = (e - e_prev) / delta_t
                    e_integral = e_integral + e * delta_t
                    control_signal = e * self.Kp + e_dot * self.Kd + e_integral * self.Ki
                    current_angle[joint] = current_angle[joint] + control_signal
                    # Publish angles to robot
                    
                    ref_right.data = [current_angle['shoulder_right_pitch'], current_angle['shoulder_right_roll'], current_angle['elbow_right']]
                    ref_left.data = [current_angle['shoulder_left_pitch'], current_angle['shoulder_left_roll'], current_angle['elbow_left']]
                    ref_head.data = [0, current_angle['neck_pitch']]
                        
                    imitation_executor.right_pub.publish(ref_right)
                    imitation_executor.left_pub.publish(ref_left)
                    imitation_executor.head_pub.publish(ref_head)
                    # update stored data for next iteration
                    # print(i, ' : ', joint, ' : ', e)
                    e_prev = e

                    iter_pid += 1
                    #read angles from robot and save at self.instant_positions
                    self.joint_state_sub = rospy.Subscriber('/qt_robot/joints/state', JointState, self.qt_joint_state_cb)
                    while self.instant_positions['elbow_right'] == 0:
                        nothing += 1
                    current_angle[joint] = copy.copy(self.instant_positions[joint])
                    self.instant_positions['elbow_right'] = 0
                    e = self.desired_angle[joint] - current_angle[joint]
            i += 1
            self.assignment(current_angle)
            #print(current_angle)
        self.save_graphs('Robot angles vs. PID online ', self.robot_angles, self.estimated_angles, 'Robot calculated signal', 'Robot PID signal')
    
    def read_skeleton_joints(self):
        '''
        This function reads the values of the skeleton points from a csv file
        '''
        path = './csv/' + str(self.participant) + '/' + str(self.participant) + '.csv'
        df = pd.read_csv(path, on_bad_lines = 'skip', float_precision = 'round_trip')
        functions = DataProcessing()
        self.head, self.neck, self.collar, self.r_shoulder, self.r_elbow, self.r_hand, self.l_shoulder, self.l_elbow, self.l_hand, self.time = functions.get_positions_of_joints(df, self.action)
        self.calculated_angles = functions.calculate_all_angles(df, self.action)

    def filter_data(self):
        '''
        This function filters the signals with a median-filter
        '''
        filters = Filters()
        self.filtered_angles['neck_pitch'] = filters.median_filter(self.calculated_angles['neck_pitch'].copy())
        self.filtered_angles['elbow_right'] = filters.median_filter(self.calculated_angles['elbow_right'].copy())
        self.filtered_angles['elbow_left'] = filters.median_filter(self.calculated_angles['elbow_left'].copy())
        self.filtered_angles['shoulder_right_roll'] = filters.median_filter(self.calculated_angles['shoulder_right_roll'].copy())
        self.filtered_angles['shoulder_right_pitch'] = filters.median_filter(self.calculated_angles['shoulder_right_pitch'].copy())
        self.filtered_angles['shoulder_left_roll'] = filters.median_filter(self.calculated_angles['shoulder_left_roll'].copy())
        self.filtered_angles['shoulder_left_pitch'] = filters.median_filter(self.calculated_angles['shoulder_left_pitch'].copy())
        # self.save_graphs('Original vs. Filtered ', self.calculated_angles, self.filtered_angles, 'Skeleton original data', 'Skeleton filtered data')

    def save_graphs(self, name, signal_1, signal_2, label_1, label_2):
        '''
        This function saves/plots 2 signals in a single graph
        '''
        time = np.arange(0, len(signal_1['neck_pitch']), 1) 
        ploting = DataProcessing()
        ploting.plot_two_signals(signal_1['neck_pitch'], signal_2['neck_pitch'], time, 'Neck pitch - ' + name, self.action, self.participant, label_1, label_2)
        ploting.plot_two_signals(signal_1['elbow_right'], signal_2['elbow_right'], time, 'Elbow right - ' + name, self.action, self.participant, label_1, label_2)
        ploting.plot_two_signals(signal_1['elbow_left'], signal_2['elbow_left'], time, 'Elbow left - ' + name, self.action, self.participant, label_1, label_2)
        ploting.plot_two_signals(signal_1['shoulder_right_roll'], signal_2['shoulder_right_roll'], time, 'Shoulder roll right - ' + name, self.action, self.participant, label_1, label_2)
        ploting.plot_two_signals(signal_1['shoulder_right_pitch'], signal_2['shoulder_right_pitch'], time, 'Shoulder pitch right - ' + name, self.action, self.participant, label_1, label_2)
        ploting.plot_two_signals(signal_1['shoulder_left_roll'], signal_2['shoulder_left_roll'], time, 'Shoulder roll left  - ' + name, self.action, self.participant, label_1, label_2)        
        ploting.plot_two_signals(signal_1['shoulder_left_pitch'], signal_2['shoulder_left_pitch'], time, 'Shoulder pitch left - ' + name, self.action, self.participant, label_1, label_2)

    def check_possible_collitions(self):
        '''
        This function verifies all possible configurations to find collisions 
        in case of collision recalculates the values of the angles
        right_elbow ---------> 0
        left_elbow ----------> 1
        right_shoulder_pitch-> 2
        left_shoulder_pitch -> 3
        right_shoulder_roll -> 4
        left_shoulder_roll --> 5
        neck_pitch ----------> 6
        neck_roll -----------> 7
        '''
        data_matrix = []
        avoidance = SelfcollitionAvoidanceModule()
        data_matrix.append(self.filtered_angles['elbow_right'].copy())
        data_matrix.append(self.filtered_angles['elbow_left'].copy())
        data_matrix.append(self.filtered_angles['shoulder_right_pitch'].copy())
        data_matrix.append(self.filtered_angles['shoulder_left_pitch'].copy())
        data_matrix.append(self.filtered_angles['shoulder_right_roll'].copy())
        data_matrix.append(self.filtered_angles['shoulder_left_roll'].copy())
        data_matrix.append(self.filtered_angles['neck_pitch'].copy())
        data_matrix = avoidance.check_all_angles(data_matrix.copy())
        self.compensated_angles['elbow_right'] = data_matrix[0]
        self.compensated_angles['elbow_left'] = data_matrix[1]
        self.compensated_angles['shoulder_right_pitch'] = data_matrix[2]
        self.compensated_angles['shoulder_left_pitch'] = data_matrix[3]
        self.compensated_angles['shoulder_right_roll'] = data_matrix[4]
        self.compensated_angles['shoulder_left_roll'] = data_matrix[5]
        self.compensated_angles['neck_pitch'] = data_matrix[6]
        # self.save_graphs('Filtered vs. Compensated ', self.filtered_angles, self.compensated_angles, 'Skeleton filtered data', 'Skeleton compensated data')

    def transform_to_robot_angles(self):
        '''
        This function performs the transformation process from skeleton angles to robot angles
        '''
        functions = DataProcessing()
        self.robot_angles['neck_pitch'] = functions.from_skeleton_to_robot_angle(self.compensated_angles['neck_pitch'].copy(), 'neck_pitch')
        self.robot_angles['elbow_left'] = functions.from_skeleton_to_robot_angle(self.compensated_angles['elbow_left'].copy(), 'elbow_left')
        self.robot_angles['elbow_right'] = functions.from_skeleton_to_robot_angle(self.compensated_angles['elbow_right'].copy(), 'elbow_right')
        self.robot_angles['shoulder_left_pitch'] = functions.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_left_pitch'].copy(), 'shoulder_left')
        self.robot_angles['shoulder_left_roll'] = functions.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_left_roll'].copy(), 'shoulder_roll')
        self.robot_angles['shoulder_right_pitch'] = functions.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_right_pitch'].copy(), 'shoulder_right')
        self.robot_angles['shoulder_right_roll'] = functions.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_right_roll'].copy(), 'shoulder_roll')
        
    def execute_online(self):   
        # wait for publisher/subscriber connections
        '''
        This function carry out the process from start to finish
        initially read the skeleton points
        at the end save the sequences calculated by the robot in csv documents
        ''' 
        wtime_begin = rospy.get_time()
        while (imitation_executor.right_pub.get_num_connections() == 0):
            rospy.loginfo("waiting for subscriber connections...")
            if rospy.get_time() - wtime_begin > 10.0:
                rospy.logerr("Timeout while waiting for subscribers connection!")
                sys.exit()
            rospy.sleep(1)
        rospy.loginfo("publishing motor command...")
        
        try:
            self.PID_online()
            # self.PID()
        except rospy.ROSInterruptException:
            rospy.logerr("could not publish motor command!")

        rospy.loginfo("motor command published")

    def save_target_data(self, idx):
        '''
        This function saves all the data of the robot generated by the PID controller
        '''
        current_time = rospy.Time.now().to_sec()
        self.target_positions['shoulder_right_pitch'].append([current_time, self.robot_angles['shoulder_right_pitch'][idx]])
        self.target_positions['shoulder_right_roll'].append([current_time,  self.robot_angles['shoulder_right_roll'][idx]])
        self.target_positions['shoulder_left_pitch'].append([current_time,  self.robot_angles['shoulder_left_pitch'][idx]])
        self.target_positions['shoulder_left_roll'].append([current_time,  self.robot_angles['shoulder_left_roll'][idx]])
        self.target_positions['neck_pitch'].append([current_time,  self.robot_angles['neck_pitch'][idx]])
        self.target_positions['elbow_right'].append([current_time,  self.robot_angles['elbow_right'][idx]])
        self.target_positions['elbow_left'].append([current_time,  self.robot_angles['elbow_left'][idx]])
    
    def save_recordings(self):
        '''
        This function saves the angles in a csv file to reproduce the actions after
        '''
        path = './csv/' + str(self.participant) + '/' + self.action + '/' + str(self.participant) + '-' + self.action + '.data'
        with open(path, 'wb') as f:
            joblib.dump([self.target_positions, self.recorded_positions], f, protocol = 2)

    def save_move(self):
        path = './csv/' + str(self.participant) + '/' + self.action + '/' + str(self.participant) + '-' + self.action + ' - '
        saver = DataProcessing()
        saver.save_data_for_qt(self.estimated_angles['neck_pitch'], path + 'neck_pitch' + '.csv')
        saver.save_data_for_qt(self.estimated_angles['elbow_right'], path + 'elbow_right' + '.csv')
        saver.save_data_for_qt(self.estimated_angles['elbow_left'], path + 'elbow_left' + '.csv')
        saver.save_data_for_qt(self.estimated_angles['shoulder_left_pitch'], path + 'shoulder_left_pitch' + '.csv')
        saver.save_data_for_qt(self.estimated_angles['shoulder_left_roll'], path + 'shoulder_left_roll' + '.csv')
        saver.save_data_for_qt(self.estimated_angles['shoulder_right_pitch'], path + 'shoulder_right_pitch' + '.csv')
        saver.save_data_for_qt(self.estimated_angles['shoulder_right_roll'], path + 'shoulder_right_roll' + '.csv')
    
    def reproduce_move(self, isMirror):
        '''
        This function reproduces the actions in normal configuration or in mirror configuration
        Input: isMirror -> bool
        '''
        path = './csv/' + str(self.participant) + '/' + self.action + '/' + str(self.participant) + '-' + self.action + ' - '
        shoulder_right_roll = self.read_angle_from_csv(path + 'shoulder_right_roll' + '.csv')
        shoulder_right_pitch = self.read_angle_from_csv(path + 'shoulder_right_pitch' + '.csv')
        shoulder_left_roll = self.read_angle_from_csv(path + 'shoulder_left_roll' + '.csv')
        shoulder_left_pitch = self.read_angle_from_csv(path + 'shoulder_left_pitch' + '.csv')
        neck_pitch = self.read_angle_from_csv(path + 'neck_pitch' + '.csv')
        elbow_right = self.read_angle_from_csv(path + 'elbow_right' + '.csv')
        elbow_left = self.read_angle_from_csv(path + 'elbow_left' + '.csv')

        sequence_len = len(elbow_left)
        for i in range(sequence_len):
        # wait for publisher/subscriber connections
            wtime_begin = rospy.get_time()
            while (imitation_executor.right_pub.get_num_connections() == 0) :
                rospy.loginfo("waiting for subscriber connections...")
                if rospy.get_time() - wtime_begin > 10.0:
                    rospy.logerr("Timeout while waiting for subscribers connection!")
                    sys.exit()
                rospy.sleep(1)
            rospy.loginfo("publishing motor command...")

            try:
                ref_right = Float64MultiArray()
                ref_left = Float64MultiArray()
                ref_head = Float64MultiArray()
                if isMirror:
                    ref_left.data = [-shoulder_right_pitch[i], shoulder_right_roll[i], elbow_right[i]]
                    ref_right.data = [shoulder_left_pitch[i], shoulder_left_roll[i], elbow_left[i]]
                else:
                    ref_right.data = [shoulder_right_pitch[i], shoulder_right_roll[i], elbow_right[i]]
                    ref_left.data = [shoulder_left_pitch[i], shoulder_left_roll[i], elbow_left[i]]
                ref_head.data = [0, neck_pitch[i]]
                imitation_executor.right_pub.publish(ref_right)
                imitation_executor.left_pub.publish(ref_left)
                imitation_executor.head_pub.publish(ref_head)
                rospy.sleep(0.07)

            except rospy.ROSInterruptException:
                rospy.logerr("could not publish motor command!")
            rospy.loginfo("motor command published")

if __name__ == '__main__':
    rospy.init_node('publish_angles')
    rospy.loginfo("started!")
    participants = np.arange(1, 2, 1)
    # actions = ['teacup', 'teapot', 'spoon', 'ladle', 'shallow_plate', 'dinner_plate', 'knife', 'fork', 'salt_shaker', 'sugar_bowl','mixer', 'pressure_cooker']
    #actions = ['teacup', 'teapot', 'spoon', 'knife', 'salt_shaker', 'mixer']
    actions = ['mixer']
    for participant in participants:
        for action in actions:
            print(participant, ': ', action)
            imitation_executor = QTImitationExecutor(participant, action)
            if len(imitation_executor.time) != 0: 
                # imitation_executor.filter_data()
                # imitation_executor.check_possible_collitions()
                # imitation_executor.transform_to_robot_angles()
                # imitation_executor.execute_online()
                # imitation_executor.save_recordings()
                # imitation_executor.save_move()
                imitation_executor.reproduce_move(0)

                #13
                #16
                #1
                #12
                #19
                #1, 11
