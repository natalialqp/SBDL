import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import *
from sympy import Point3D, Line3D
import math
from homogeneous_transformation import HT

class SelfcollitionAvoidanceModule(object):
    '''
    This class belongs to the collision module, the main function is to predict collisions and recalculate the angular positions to avoid such collisions.
    Initially, the parameters of the volumes that represent QTrobot are defined
    '''
    def __init__(self):
        self.dis_from_origin_ellipse = [0, 0, 0.52]  
        self.ellipse_coefs = [0.10, 0.12, 0.09]  
        self.dis_from_origin_cone = [0, 0, 0.27]  
        self.cone_coefs = [0.08, 0.3]   
        self.dis_from_origin_sphere = [0, 0, 0.27]                                    
        self.sphere_coefs = 0.12
        self.dis_from_origin_cylinder_left = [0, -0.08, 0]                       
        self.cylinder_coefs = [0.08, 0.27]   
        self.dis_from_origin_cylinder_right = [0, 0.08, 0]   
        self.u = np.linspace(0, 2 * np.pi, 100)
        self.v = np.linspace(0, np.pi, 100)
            
        self.x_ellipse, self.y_ellipse, self.z_ellipse = self.generate_ellipse()
        self.x_cone, self.y_cone, self.z_cone = self.generate_cone()
        self.x_sphere, self.y_sphere, self.z_sphere = self.generate_sphere()
        self.x_cylinder_left, self.y_cylinder_left, self.z_cylinder_left = self.generate_cylinder(self.dis_from_origin_cylinder_left)
        self.x_cylinder_right, self.y_cylinder_right, self.z_cylinder_left = self.generate_cylinder(self.dis_from_origin_cylinder_right)
        
        self.last_safe_angles = np.zeros((8))
        
    def generate_ellipse(self):
        '''
        This function generates the volume of a ellipse
        '''
        #Ellipse
        x_ellipse = self.ellipse_coefs[0] * np.outer(np.cos(self.u), np.sin(self.v)) + self.dis_from_origin_ellipse[0]
        y_ellipse = self.ellipse_coefs[1] * np.outer(np.sin(self.u), np.sin(self.v)) + self.dis_from_origin_ellipse[1]
        z_ellipse = self.ellipse_coefs[2] * np.outer(np.ones_like(self.u), np.cos(self.v)) + self.dis_from_origin_ellipse[2]
        return x_ellipse, y_ellipse, z_ellipse
    
    def check_ellipse(self, end_point):
        '''
        This function verifies if a point lies inside the ellipse
        Input: point to be checked
        Output: inside -> bool
        '''
        #Ellipse equation 
        inside = False
        ellipse_eq = (((end_point[0] - self.dis_from_origin_ellipse[0])/self.ellipse_coefs[0])**2
                    + ((end_point[1] - self.dis_from_origin_ellipse[1])/self.ellipse_coefs[1])**2 
                    + ((end_point[2] - self.dis_from_origin_ellipse[2])/self.ellipse_coefs[2])**2) 
        if ellipse_eq <= 1:
            inside = True
        return inside
    
    def mirror_point_ellipse(self, end_point):
        '''
        This function projects outside the ellipse to a point inside
        Input: end_point -> point to be checked
        Output: end_point -> proposed point
        '''
        new_x = (self.ellipse_coefs[0] * np.sqrt(1
                - ((end_point[1] - self.dis_from_origin_ellipse[1])/self.ellipse_coefs[1])**2 
                - ((end_point[2] - self.dis_from_origin_ellipse[2])/self.ellipse_coefs[2])**2) 
                + self.dis_from_origin_ellipse[0])*1.15
        new_point = [new_x, end_point[1], end_point[2]]
        return new_point
    
    def generate_cone(self):
        '''
        This function generates the volume of a cone
        '''
        #Cone
        alpha = np.arctan(self.cone_coefs[0]/self.cone_coefs[1])                                                                            
        r_1 = np.linspace(0, self.cone_coefs[0], 100)
        theta_1, R_1 = np.meshgrid(self.u, r_1)
        x_cone = R_1 * np.cos(theta_1) + self.dis_from_origin_cone[0]
        y_cone = R_1 * np.sin(theta_1) + self.dis_from_origin_cone[0]
        z_cone = -(R_1/self.cone_coefs[0]) * self.cone_coefs[1] + self.cone_coefs[1] + self.dis_from_origin_cone[2]
        return x_cone, y_cone, z_cone
    
    def check_cone(self, end_point):
        '''
        This function verifies if a point lies inside the cone
        Input: point to be checked
        Output: inside -> bool
        '''
        #Cone equation 
        inside = False
        alpha = np.arctan(self.cone_coefs[0]/self.cone_coefs[1])                                                                            
        alpha_point = np.arctan((np.sqrt(end_point[0]**2 + end_point[1]**2))/
                      (self.cone_coefs[1] + self.dis_from_origin_cone[2] - end_point[2]))
        if (alpha >= alpha_point and end_point[2] <= self.cone_coefs[1] 
        + self.dis_from_origin_cone[2] and end_point[2] >= self.dis_from_origin_cone[2]):
            inside = True
        return inside
        
    def mirror_point_cone(self, end_point):
        '''
        This function projects outside the cone to a point inside
        Input: end_point -> point to be checked
        Output: end_point -> proposed point
        '''
        alpha = np.arctan(self.cone_coefs[0]/self.cone_coefs[1]) 
        new_x = np.sqrt(((self.cone_coefs[1] + self.dis_from_origin_cone[2] - end_point[2])*np.tan(alpha))**2 
                        - end_point[1]**2)*1.15
        new_point = [new_x, end_point[1], end_point[2]]
        return new_point
    
    def generate_sphere(self):
        '''
        This function generates the volume of a sphere
        '''
        #Sphere              
        x_sphere = self.sphere_coefs * np.outer(np.cos(self.u), np.sin(self.v)) + self.dis_from_origin_sphere[0]
        y_sphere = self.sphere_coefs * np.outer(np.sin(self.u), np.sin(self.v)) + self.dis_from_origin_sphere[1]
        z_sphere = self.sphere_coefs * np.outer(np.ones_like(self.u), np.cos(self.v)) + self.dis_from_origin_sphere[2]
        return x_sphere, y_sphere, z_sphere
        
    def check_sphere(self, end_point):
        '''
        This function verifies if a point lies inside the sphere
        Input: point to be checked
        Output: inside -> bool
        '''
        #Sphere equation 
        inside = False
        sphere_eq = ((end_point[0] - self.dis_from_origin_sphere[0])**2 
                    + (end_point[1] - self.dis_from_origin_sphere[1])**2 
                    + (end_point[2] - self.dis_from_origin_sphere[2])**2)
        if np.sqrt(sphere_eq) <= self.sphere_coefs:
            inside = True
        return inside

    def mirror_point_sphere(self, end_point):
        '''
        This function projects outside the sphere to a point inside
        Input: end_point -> point to be checked
        Output: end_point -> proposed point
        '''
        new_x = (np.sqrt((self.sphere_coefs*1.1)**2
                - (end_point[1] - self.dis_from_origin_sphere[1])**2 
                - (end_point[2] - self.dis_from_origin_sphere[2])**2) 
                + self.dis_from_origin_sphere[0])*1.15
        new_point = [new_x, end_point[1], end_point[2]]
        return new_point
    
    def generate_cylinder(self, dis_from_origin_cylinder):     
        '''
        This function generates the volume of a cylinder
        '''                                    
        z_cylinder = np.linspace(0, self.cylinder_coefs[1], 100)
        theta, z_cylinder = np.meshgrid(self.u, z_cylinder)
        x_cylinder = self.cylinder_coefs[0] * np.cos(theta) + dis_from_origin_cylinder[0]
        y_cylinder = self.cylinder_coefs[0] * np.sin(theta) + dis_from_origin_cylinder[1]
        return x_cylinder, y_cylinder, z_cylinder
       
    def check_cylinder(self, dis_from_origin_cylinder, end_point): 
        '''
        This function verifies if a point lies inside the cylinder
        Input: point to be checked
        Output: inside -> bool
        '''   
        inside = False
        cylinder_equation = ((end_point[0] - dis_from_origin_cylinder[0])**2 
                           + (end_point[1] - dis_from_origin_cylinder[1])**2)
        if np.sqrt(cylinder_equation) <= self.cylinder_coefs[0] and end_point[2] <= self.cylinder_coefs[1]:
            inside = True
        return inside
    
    def mirror_point_cylinder(self, end_point, dis_from_origin_cylinder):
        '''
        This function projects outside the cylinder to a point inside
        Input: end_point -> point to be checked
        Output: end_point -> proposed point
        '''
        new_x = (np.sqrt(self.cylinder_coefs[0]**2
                - (end_point[1] - dis_from_origin_cylinder[1])**2)
                + dis_from_origin_cylinder[0])*1.15
        new_point = [new_x, end_point[1], end_point[2]]
        return new_point
    
    def check_all_volumes(self, end_point):
        '''
        This function calls all volumes to verify the point
        Input: end_point -> point to be checked
        Output: end_point -> proposed point
        '''
        if(self.check_ellipse(end_point)):
            end_point = self.mirror_point_ellipse(end_point.copy())
        if(self.check_cone(end_point)):
            end_point = self.mirror_point_cone(end_point.copy())
        if(self.check_sphere(end_point)):
            end_point = self.mirror_point_sphere(end_point.copy())
        if(self.check_cylinder(end_point, self.dis_from_origin_cylinder_left)):
            end_point = self.mirror_point_cylinder(end_point.copy(), self.dis_from_origin_cylinder_left)
        if(self.check_cylinder(end_point, self.dis_from_origin_cylinder_right)):
            end_point = self.mirror_point_cylinder(end_point.copy(), self.dis_from_origin_cylinder_right)
        return np.array(end_point)
            
    def check_all_angles(self, angles_matrix):
        '''
        This function receives all the configurations and predicts possible collisions for all the points
        Input: angles_matrix from skeleton
        Output: angles_matrix predicted
        '''
        for i in range(len(angles_matrix[0])):
            angles = np.array([angles_matrix[0][i], angles_matrix[1][i], angles_matrix[2][i], angles_matrix[3][i], 
                               angles_matrix[4][i], angles_matrix[5][i], angles_matrix[6][i], 0])
            angles = self.predict_collition(angles)
            angles_matrix[0][i] = angles[0] 
            angles_matrix[1][i] = angles[1] 
            angles_matrix[2][i] = angles[2] 
            angles_matrix[3][i] = angles[3] 
            angles_matrix[4][i] = angles[4] 
            angles_matrix[5][i] = angles[5] 
            angles_matrix[6][i] = angles[6]
        return angles_matrix
    
    def print_model_3d(self, end_point, name): 
        '''
        This function plots/saves the end_point and the volume
        '''
        #Change the Size of Graph using Figsize
        fig = plt.figure(figsize = (15, 15))
        #Generating a 3D sine wave
        ax = plt.axes(projection = '3d')

        #plot 
        ax.plot_wireframe(self.x_ellipse, self.y_ellipse, self.z_ellipse, rstride = 4, cstride = 4, color = 'b')
        ax.plot_wireframe(self.x_cone, self.y_cone, self.z_cone, rstride = 4, cstride = 4, color = 'y')
        ax.plot_wireframe(self.x_sphere, self.y_sphere, self.z_sphere, rstride = 4, cstride = 4, color = 'g')
        ax.plot_wireframe(self.x_cylinder_left, self.y_cylinder_left, self.z_cylinder_left, rstride = 4, cstride = 4, color = 'y')
        ax.plot_wireframe(self.x_cylinder_right, self.y_cylinder_right, self.z_cylinder_left, rstride = 4, cstride = 4, color = 'y')
        ax.scatter(end_point[0],end_point[1], end_point[2], color = 'm')

        # Labels' name
        plt.xlabel('X - axis [m]')
        plt.ylabel('Y - axis [m]')
        ax.set_zlabel('Z - axis [m]', fontsize = 10)
        ax.auto_scale_xyz([-0.25, 0.25], [-0.25, 0.25], [0.0, 0.50])
        # trun off/on axis
        plt.axis('on')
        plt.savefig(name + '.pdf')  
        for ii in range(0, 360, 45):
                ax.view_init(elev=10., azim=ii)
                plt.savefig(name + "movie%d.pdf" % ii)
    
    def Forward_Kinematics(self, angles):
        '''
        This function receives an angle configuration and calculates the spatial position of the end effector and the elbow
        right_elbow ---------> 0
        left_elbow ----------> 1
        right_shoulder_pitch-> 2
        left_shoulder_pitch -> 3
        right_shoulder_roll -> 4
        left_shoulder_roll --> 5
        neck_pitch ----------> 6
        neck_roll -----------> 7
        '''
        Shoulder_Pitch_Left  = [0,    0.08,  0.396]
        Shoulder_Pitch_Right = [0,   -0.08,  0.396]        
        Shoulder_Roll_Left   = [0,  0.0445,      0]
        Shoulder_Roll_Right  = [0,  0.0445,      0]
        Elbow_Right          = [0, 0.07708,      0]
        Elbow_Left           = [0, 0.07708,      0]
        Hand_Right           = [0,   0.184,      0]
        Hand_Left            = [0,   0.184,      0]
        Head_Yaw             = [0,       0,  0.338]
        Head_Pitch           = [0,       0, 0.0962]
        Camera               = [0.094,   0,  0.162]

        Shoulder_Pitch_Left_angle  = [0,        0,       0]
        Shoulder_Pitch_Right_angle = [3.14159,  0,       0]

        Shoulder_Roll_Left_angle   = [-np.pi/3 + angles[5],  angles[3],       0]        # -1.5708
        Shoulder_Roll_Right_angle  = [ np.pi/3 + angles[4],  angles[2],       0]        # -1.5708

        Elbow_Right_angle          = [ np.pi/6 + angles[0],        0,       0]
        Elbow_Left_angle           = [-np.pi/6 - angles[1],        0,       0]

        Hand_Left_angle            = [0,        -0,        0]
        Hand_Right_angle           = [0,        -0,        0]

        Head_Yaw_angle             = [0,        -0,        0]
        Head_Pitch_angle           = [0,        -0,        0]
        Camera_angle               = [-np.pi/2,  0, -np.pi/2]
        a = HT()
        B_T_LSP   = a.get_homogeneous_transform(Shoulder_Pitch_Left_angle, Shoulder_Pitch_Left)
        LSP_T_LSR = a.get_homogeneous_transform(Shoulder_Roll_Left_angle, Shoulder_Roll_Left)
        LSR_T_LER = a.get_homogeneous_transform(Elbow_Left_angle, Elbow_Left)
        LER_T_LH  = a.get_homogeneous_transform(Hand_Left_angle, Hand_Left)

        B_T_RSP   = a.get_homogeneous_transform(Shoulder_Pitch_Right_angle, Shoulder_Pitch_Right)
        RSP_T_RSR = a.get_homogeneous_transform(Shoulder_Roll_Right_angle, Shoulder_Roll_Right)
        RSR_T_RER = a.get_homogeneous_transform(Elbow_Right_angle, Elbow_Right)
        RER_T_RH  = a.get_homogeneous_transform(Hand_Right_angle, Hand_Right)

        B_T_LSR = B_T_LSP.dot(LSP_T_LSR)
        B_T_LER = B_T_LSR.dot(LSR_T_LER)
        B_T_LH = B_T_LER.dot(LER_T_LH)

        B_T_RSR = B_T_RSP.dot(RSP_T_RSR)
        B_T_RER = B_T_RSR.dot(RSR_T_RER)
        B_T_RH = B_T_RER.dot(RER_T_RH)

        left_end_effector_pos = a.get_translation(B_T_LH)
        left_elbow_pos = a.get_translation(B_T_LER)
        left_shoulder_pos = a.get_translation(B_T_LSR)

        right_end_effector_pos = a.get_translation(B_T_RH)
        right_elbow_pos = a.get_translation(B_T_RER)
        right_shoulder_pos = a.get_translation(B_T_RSR)

        return left_end_effector_pos, left_elbow_pos, left_shoulder_pos, right_end_effector_pos, right_elbow_pos, right_shoulder_pos

    def calculate_coeff(self, l1, l2, z, xn, yn, zn, xs, ys, zs):
        '''
        This function computes all possible values that satisfy the inverse kinematics equations
        Input:
        l1 -> distance between shoulder and elbow
        l2 -> distance between elbow and hand
        z -> z position of elbow
        xn -> new x position of hand
        yn -> new y position of hand
        zn -> new z position of hand
        xs -> x position of shoulder
        ys -> y position of shoulder
        zs -> z position of shoulder

        Output:
        vec -> new point 
        '''
        x, y = symbols('x, y')
        eq1 = x**2 + (y - ys)**2 + (z - zs)**2 - l1**2
        eq2 = (x - xn)**2 + (y - yn)**2 + (z - zn)**2 - l2**2
        A = sp.Matrix([eq1, eq2])
        res = sp.solve(A, [x, y])
        vec = np.zeros((3))
        for i in res:
            if isinstance(i, complex):
                if i[0] > 0:
                    vec = [i[0], i[1], z]
        return vec

    def Invers_Kinematics(self, new_hand_pose, pos_hand, pos_elbow, pos_shoulder, hand):
        '''
        This function computes the new angle configuration for the new proposed point to avoid collision
        Input:
        new_hand_pose -> new proposed point
        pos_hand -> previous hand point
        pos_elbow -> previous elbow point
        pos_shoulder -> previous shoulder point
        hand -> left or right

        Output:
        angle_elbow -> new angle for the elbow
        angle_shoulder_roll-> new angle for the shoulder roll
        angle_shoulder_pitch -> new angle for the shoulder pitch
        '''
        l_1 = np.linalg.norm(pos_shoulder - pos_elbow)
        l_2 = np.linalg.norm(pos_hand - pos_elbow)
        l_3_new = np.linalg.norm(pos_shoulder - new_hand_pose)
        new_elbow = self.calculate_coeff(l_1, l_2, pos_elbow[2], new_hand_pose[0], new_hand_pose[1], new_hand_pose[2], pos_shoulder[0], pos_shoulder[1], pos_shoulder[2])
        if all(new_elbow) == 0:
            if hand == 'left':
                angle_elbow = self.last_safe_angles[1]
                angle_shoulder_roll = self.last_safe_angles[5]
                angle_shoulder_pitch = self.last_safe_angles[3]
            else: 
                angle_elbow = self.last_safe_angles[0]
                angle_shoulder_roll = self.last_safe_angles[4]
                angle_shoulder_pitch = self.last_safe_angles[2]
        else:
            aux_angle = (l_1**2 + l_2**2 - l_3_new**2)/(2*l_1*l_2)
            if aux_angle > 1:
                aux_angle = 1
            elif aux_angle < -1:
                aux_angle = -1
            angle_elbow = np.rad2deg(math.acos(aux_angle))
            vec = new_elbow - pos_shoulder
            angle_shoulder_pitch = np.rad2deg(math.atan2(vec[0], vec[2]))
            angle_shoulder_roll = np.rad2deg(math.atan2(vec[1], vec[2]))
        return angle_elbow, angle_shoulder_roll, angle_shoulder_pitch

    def predict_collition(self, angles): 
        '''
        This is the main function to execute, it contains all the procedure to predict collision
        Input -> angles
        right_elbow ---------> 0
        left_elbow ----------> 1 
        right_shoulder_pitch-> 2 
        left_shoulder_pitch -> 3 
        right_shoulder_roll -> 4
        left_shoulder_roll --> 5
        neck_pitch ----------> 6
        neck_roll -----------> 7 
        Output -> new proposed angles
       '''    
        left_hand, left_elbow, left_shoulder, right_hand, right_elbow, right_shoulder = self.Forward_Kinematics(angles)
        new_left_hand = self.check_all_volumes(left_hand)
        new_right_hand = self.check_all_volumes(right_hand)
        if (left_hand != new_left_hand).any():
            max_elbow_left, max_shoulder_left_roll, max_shoulder_left_pitch = self.Invers_Kinematics(new_left_hand, left_hand, left_elbow, left_shoulder, 'left')
            angles[1] = max_elbow_left
            angles[3] = max_shoulder_left_pitch
            angles[5] = max_shoulder_left_roll
        else: 
            self.last_safe_angles[1] = angles[1]
            self.last_safe_angles[3] = angles[3]
            self.last_safe_angles[5] = angles[5]
            
        if (right_hand != new_right_hand).any():
            max_elbow_right, max_shoulder_right_roll, max_shoulder_right_pitch = self.Invers_Kinematics(new_right_hand, right_hand, right_elbow, right_shoulder, 'right')
            angles[0] = max_elbow_right
            angles[2] = max_shoulder_right_pitch
            angles[4] = max_shoulder_right_roll
        else: 
            self.last_safe_angles[0] = angles[0]
            self.last_safe_angles[2] = angles[2]
            self.last_safe_angles[4] = angles[4]
        return angles
