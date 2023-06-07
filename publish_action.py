#!/usr/bin/env python3
import sys
from telnetlib import PRAGMA_HEARTBEAT
import rospy
import pandas as pd
import joblib
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import csv
import data_filtering
import collition_avoidance
import data_processing

def read_angle_from_csv(name):
    angle_vec = []
    with open(name, 'r') as file:                                                               #'./src/control_angles/user_01/spoon_01/' 
        reader = csv.reader(file)
        for row in reader:
            angle_vec.append(float(row[0])) 
    return angle_vec

class QTImitationExecutor(object):

    def __init__(self):
        # self.target_positions = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
        #                          'shoulder_left_pitch': [], 'shoulder_left_roll': [],
        #                          'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.calculated_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
                                 'shoulder_left_pitch': [], 'shoulder_left_roll': [],
                                 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.filtered_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
                                 'shoulder_left_pitch': [], 'shoulder_left_roll': [],
                                 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.recorded_positions = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
                                   'shoulder_left_pitch': [], 'shoulder_left_roll': [],
                                   'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.estimated_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
                                'shoulder_left_pitch': [], 'shoulder_left_roll': [],
                                'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.compensated_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
                                 'shoulder_left_pitch': [], 'shoulder_left_roll': [],
                                 'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.robot_angles = {'shoulder_right_pitch': [], 'shoulder_right_roll': [],
                             'shoulder_left_pitch': [], 'shoulder_left_roll': [],
                             'neck_pitch': [], 'elbow_right': [], 'elbow_left': []}
        self.instant_positions = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,
                                  'shoulder_left_pitch': 0, 'shoulder_left_roll': 0,
                                  'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}             
        self.desired_angle = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,
                              'shoulder_left_pitch': 0, 'shoulder_left_roll': 0,
                              'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        self.joint_names = ['shoulder_right_pitch', 'shoulder_right_roll',
                            'shoulder_left_pitch', 'shoulder_left_roll',
                            'neck_pitch', 'elbow_right', 'elbow_left']    
        self.Kp = 0.8
        self.Ki = 0.1
        self.Kd = 0.001
        self.read_skeleton_joints()
        self.right_pub = rospy.Publisher('/qt_robot/right_arm_position/command', Float64MultiArray, queue_size = 1)
        self.left_pub = rospy.Publisher('/qt_robot/left_arm_position/command', Float64MultiArray, queue_size = 1)
        self.head_pub = rospy.Publisher('/qt_robot/head_position/command', Float64MultiArray, queue_size = 1)


    def addition(self, dict_1, dict_2):
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict_1[i] + dict_2[i]
        return dict_res

    def division(self, dict, num):
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict[i]/ num
        return dict_res

    def substraction(self, dict_1, dict_2):
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict_1[i] - dict_2[i] 
        return dict_res

    def multiplication(self, dict, num):
        dict_res = {'shoulder_right_pitch': 0, 'shoulder_right_roll': 0,'shoulder_left_pitch': 0, 'shoulder_left_roll': 0, 'neck_pitch': 0, 'elbow_right': 0, 'elbow_left': 0}
        for i in self.joint_names:
            dict_res[i] = dict[i] * num 
        return dict_res

    def assignment(self, dict):
        for i in self.joint_names:
            self.estimated_angles[i].append(dict[i])
    
    # def save_target_data(self, idx):
    #     current_time = rospy.Time.now().to_sec()
    #     self.target_positions['shoulder_right_pitch'].append([current_time, self.shoulder_right_pitch[idx]])
    #     self.target_positions['shoulder_right_roll'].append([current_time, self.shoulder_right_roll[idx]])
    #     self.target_positions['shoulder_left_pitch'].append([current_time, self.shoulder_left_pitch[idx]])
    #     self.target_positions['shoulder_left_roll'].append([current_time, self.shoulder_left_roll[idx]])
    #     self.target_positions['neck_pitch'].append([current_time, self.neck_pitch[idx]])
    #     self.target_positions['elbow_right'].append([current_time, self.elbow_right[idx]])
    #     self.target_positions['elbow_left'].append([current_time, self.elbow_left[idx]])

    def qt_joint_state_cb(self, joint_state_msg):
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

    def execute(self):
        sequence_len = len(self.elbow_left)
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
                print("TRIAL: ", i)

                ref_right = Float64MultiArray()
                ref_left = Float64MultiArray()
                ref_head = Float64MultiArray()

                ref_right.data = [self.shoulder_right_pitch[i], self.shoulder_right_roll[i], self.elbow_right[i]]
                ref_left.data = [self.shoulder_left_pitch[i], self.shoulder_left_roll[i], self.elbow_left[i]]
                ref_head.data = [0, self.neck_pitch[i]]

                # self.save_target_data(i)

                imitation_executor.right_pub.publish(ref_right)
                imitation_executor.left_pub.publish(ref_left)
                imitation_executor.head_pub.publish(ref_head)
                rospy.sleep(0.5)

            except rospy.ROSInterruptException:
                rospy.logerr("could not publish motor command!")

            rospy.loginfo("motor command published")

    def PID(self):
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
            
            #read angles from robot and save at self.instant_positions
            self.joint_state_sub = rospy.Subscriber('/qt_robot/joints/state', JointState, self.qt_joint_state_cb)
            while self.instant_positions['elbow_right'] == 0:
                nothing += 1
            current_angle = self.instant_positions.copy()  
            e = self.substraction(self.desired_angle, current_angle)
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
        return current_angle

    def read_skeleton_joints(self):
        path = './csv/' + str(self.participant) + '/' + str(self.participant) + '.csv'
        df = pd.read_csv(path, on_bad_lines = 'skip', float_precision = 'round_trip')
        self.head, self.neck, self.collar, self.r_shoulder, self.r_elbow, self.r_hand, self.l_shoulder, self.l_elbow, self.l_hand, self.time = data_processing.get_positions_of_joints(df, self.participant, self.action)
        self.calculated_angles = data_processing.calculate_all_angles(self, df, self.action, self.participant)

    def filter_data(self):
        self.filtered_angles['neck_pitch'] = data_filtering.low_pass_filter(self.calculated_angles['neck_pitch'])
        self.filtered_angles['elbow_right'] = data_filtering.low_pass_filter(self.calculated_angles['elbow_right'])
        self.filtered_angles['elbow_left'] = data_filtering.low_pass_filter(self.calculated_angles['elbow_left'])
        self.filtered_angles['shoulder_right_roll'] = data_filtering.low_pass_filter(self.calculated_angles['shoulder_right_roll'])
        self.filtered_angles['shoulder_right_pitch'] = data_filtering.low_pass_filter(self.calculated_angles['shoulder_right_pitch'])
        self.filtered_angles['shoulder_left_roll'] = data_filtering.low_pass_filter(self.calculated_angles['shoulder_left_roll'])
        self.filtered_angles['shoulder_left_pitch'] = data_filtering.low_pass_filter(self.calculated_angles['shoulder_left_pitch'])

    def check_possible_collitions(self):
        '''right_elbow ---------> 0
           left_elbow ----------> 1
           right_shoulder_pitch-> 2
           left_shoulder_pitch -> 3
           right_shoulder_roll -> 4
           left_shoulder_roll --> 5
           neck_pitch ----------> 6
           neck_roll -----------> 7
        '''
        data_matrix = []
        data_matrix.append(self.filtered_angles['elbow_right'])
        data_matrix.append(self.filtered_angles['elbow_left'])
        data_matrix.append(self.filtered_angles['shoulder_right_pitch'])
        data_matrix.append(self.filtered_angles['shoulder_left_pitch'])
        data_matrix.append(self.filtered_angles['shoulder_right_roll'])
        data_matrix.append(self.filtered_angles['shoulder_left_roll'])
        data_matrix.append(self.filtered_angles['neck_pitch'])
        data_matrix = collition_avoidance.check_all_angles(data_matrix)
        self.compensated_angles['elbow_right'] = data_matrix[0]
        self.compensated_angles['elbow_left'] = data_matrix[1]
        self.compensated_angles['shoulder_right_pitch'] = data_matrix[2]
        self.compensated_angles['shoulder_left_pitch'] = data_matrix[3]
        self.compensated_angles['shoulder_right_roll'] = data_matrix[4]
        self.compensated_angles['shoulder_left_roll'] = data_matrix[5]
        self.compensated_angles['neck_pitch'] = data_matrix[6]

    def transform_to_robot_angles(self):
        self.robot_angles['neck_pitch'] = data_processing.from_skeleton_to_robot_angle(self.compensated_angles['neck_pitch'], 'neck_pitch')
        self.robot_angles['elbow_left'] = data_processing.from_skeleton_to_robot_angle(self.compensated_angles['elbow_left'], 'elbow_left')
        self.robot_angles['elbow_right'] = data_processing.from_skeleton_to_robot_angle(self.compensated_angles['elbow_right'], 'elbow_right')
        self.robot_angles['shoulder_left_pitch'] = data_processing.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_left_pitch'], 'shoulder_left')
        self.robot_angles['shoulder_left_roll'] = data_processing.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_left_roll'], 'shoulder_roll')
        self.robot_angles['shoulder_right_pitch'] = data_processing.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_right_pitch'], 'shoulder_right')
        self.robot_angles['shoulder_right_roll'] = data_processing.from_skeleton_to_robot_angle(self.compensated_angles['shoulder_right_roll'], 'shoulder_roll')
        print(self.robot_angles)
        
    def execute_online(self):   
        # wait for publisher/subscriber connections
            wtime_begin = rospy.get_time()
            while (imitation_executor.right_pub.get_num_connections() == 0):
                rospy.loginfo("waiting for subscriber connections...")
                if rospy.get_time() - wtime_begin > 10.0:
                    rospy.logerr("Timeout while waiting for subscribers connection!")
                    sys.exit()
                rospy.sleep(1)
            rospy.loginfo("publishing motor command...")
        
            try:
                self.PID()
            except rospy.ROSInterruptException:
                rospy.logerr("could not publish motor command!")

            rospy.loginfo("motor command published")

    def save_recordings(self):
        with open('./src/execution_recordings/user1_teapot_test.data', 'wb') as f:
            joblib.dump([self.target_positions, self.recorded_positions], f, protocol = 2)

if __name__ == '__main__':
    rospy.init_node('publish_angles')
    rospy.loginfo("started!")

    imitation_executor = QTImitationExecutor()
    # imitation_executor.execute()
    imitation_executor.filter_data()
    imitation_executor.check_possible_collitions()
    imitation_executor.transform_to_robot_angles()
    #imitation_executor.execute_online()
    imitation_executor.save_recordings()