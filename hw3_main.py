import numpy as np
from utils import load_data
from skimage.io import imshow
from scipy.linalg import expm, logm, inv
from numpy.linalg import multi_dot

#Shape of grid
SHAPE = (2000,2000)

#Scale K array in pose update to handle singularity issues, manually changed depending on dataset
SCALE = (10**(-4))

#variance of motion model
W = 0.01

#Variance of observation model
V = 0.01

#Projection Matrix
P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])


class KalmanSlamRobot():
    def __init__(self):
        #Occupancy grid map 
        self.grid = np.zeros(SHAPE)
        
        #value tracks position and orientation over time using INS and update from stereo
        self.invSEPose = np.array([np.identity(4)])
        #Initialize covariance matrix of pose
        self.invSEcov  = np.array([np.identity(6)])
        
        #Track small se space of robot pose, initializing at middle of map and no rotations
        self.sepose = np.array([[SHAPE[0]/2,SHAPE[1]/2,10,0,0,0]])

        #landmark and covariance array to be inistantiated 
        self.landmarks = None
        self.landCov = None
        
        self.jointCov = None
    
    #Part1: Predict IMU Pose Only
    def predictPose(self, lin, rot, delt, T):
        
        #Pose in se3 space
        sePose = np.append(lin,rot)
        
        #Pose in SE3 space
        invSEPose = expm(-1*hatMap(sePose)*delt)
        invSEcov = expm(-1*curlHat(sePose)*delt)
        
        #Change the pose mean based on new IMU data
        self.invSEPose = np.append(self.invSEPose,[invSEPose.dot(self.invSEPose[-1])], axis = 0)
        
        #Change the pose covariance based on IMU and W variable and append as new element
        self.invSEcov = np.append(self.invSEcov, [invSEcov.dot(self.invSEcov[-1].dot(invSEcov.T)) + W], axis = 0)
        
        #Keep track of se3 version of pose to update grid map easily
        self.sepose = np.append(self.sepose,[veeMap(inv(self.invSEPose[-1]))] + self.sepose[0], axis = 0)
    
    #Predict where observations might be based on landmarks in world frame and robot pose
    def predictObservations(self, T, M, matched):
        
        predLandmarks = [[0,0,0,0] for i in range(len(self.landmarks))]
        for i in matched:
                predLandmarks[i] = M.dot(project(T.dot(self.invSEPose[-1].dot(self.landmarks[i]))))
        return np.array(predLandmarks)
    
    #Part 2: Update landmarks based on IMU and EKF
    def updateLandmarks(self, feature, T, M, b):
        U = self.invSEPose[-1]
        new, matched = matchZ(feature, self.landmarks)
        zpred = self.predictObservations(T,M, matched)
        
        #Handles landmark update for each matched landmark individually
        for i in matched:
            proj = deltaProjection(multi_dot([T,U,self.landmarks[i]]))
            H = multi_dot([M,proj,T,U,P.T])
            sig = self.landCov[i]
            K = multi_dot([sig,H.T,inv(multi_dot([H,sig,H.T]) + np.identity(4)*V)])
            corr = feature[i]-zpred[i]
            self.landmarks[i] = self.landmarks[i] + np.append(K.dot(corr),0)
            self.landCov[i] = (np.identity(3) - K.dot(H)).dot(self.landCov[i])
        
        #If any new landmarks in feature that havent been seen before, instantiates landmark
        if len(new) != 0:
            self.addLandmarks(feature, new, T, M, b)
        

    #Part 3: Update Pose from Landmarks 
    def updatePose(self, feature, T, M):
        new, matched = matchZ(feature, self.landmarks)
        zpred = self.predictObservations(T,M, matched)
        U = self.invSEPose[-1]
        sig = self.invSEcov[-1]
        if len(matched) != 0:
            #Create array of jacobian values of size Nt, where Nt are all matched landmarks 
            proj = np.array([deltaProjection(multi_dot([T,U,self.landmarks[i]])) for i in matched])
    
            #H matrix for all matched features
            H = np.array([multi_dot([M,proj[i],T,identitySkew(U.dot(self.landmarks[matched[i]]))]) for i in range(len(matched))])
    
            #constant added to lower kalman gain 
            K = np.array([multi_dot([sig,H[i].T,inv(multi_dot([H[i],sig,H[i].T]) + np.identity(4)*V)]) for i in range(len(matched))])*SCALE
            
            #Reshapes H and K matrix to have dimensions 6x4N and 4Nx6 respectively
            H = H.reshape(H.shape[0]*H.shape[1],H.shape[2])
            K = collapseK(K)
            corr = (feature[matched] - zpred[matched]).flatten()
    
            #Update imu pose and variance from all lenamdark corrections
            self.invSEPose = np.append(self.invSEPose, [expm(hatMap(K.dot(corr))).dot(self.invSEPose[-1])], axis = 0)
            self.invSEcov = np.append(self.invSEcov, [(np.identity(6) - K.dot(H)).dot(sig)], axis = 0)
        
        #return corr, K, H

    
    #Adds new landmarks
    def addLandmarks(self, feature, new, T, M, b):
        fsu,cu,cv = M[0][0],M[0][2],M[1][2]
        for i in new:
            landmark = feature[i]
            disparity = landmark[0] - landmark[2]
            x = ((landmark[0]-cu)*b)/(disparity)
            y = ((landmark[1]-cv)*b)/(disparity)
            z = b*fsu/disparity
        self.landmarks[i] = inv(self.invSEPose[-1]).dot(inv(T).dot(np.array([x,y,z,1])))
    
    #Initializes landmarks in work coordinates during t=0
    def initLandmarks(self, feature, K, b, T):
        fsu,cu,cv = K[0][0],K[0][2],K[1][2]
        m = []
        for landmark in feature:
            if np.array_equal(landmark, np.array([-1,-1,-1,-1])) is False:
                disparity = landmark[0] - landmark[2]
                x = ((landmark[0]-cu)*b)/(disparity)
                y = ((landmark[1]-cv)*b)/(disparity)
                z = b*fsu/disparity
            
                m.append(inv(self.invSEPose[-1]).dot(inv(T).dot(np.array([x,y,z,1]))))
            else:
                m.append([0,0,0,0])
                
        m = np.array(m)
        
        if self.landmarks is None:
            self.landmarks = np.array(m)
            self.landCov = np.array([np.identity(3) for i in range(len(feature))])*V

#Used to reshape K in correct fashion
def collapseK(A):
    first = A[:,0].flatten()
    sec = A[:,1].flatten()
    third = A[:,2].flatten()
    fourth = A[:,3].flatten()
    fifth = A[:,4].flatten()
    sixth = A[:,5].flatten()
    
    return np.array([first,sec,third,fourth,fifth,sixth])

#Find new and matched arrays
def matchZ(feature, landmarks):
    new = []
    matched = []
    
    for i in range(len(feature)):
        if np.array_equal(feature[i], np.array([-1,-1,-1,-1])) is False and np.array_equal(landmarks[i], np.array([0,0,0,0])) is False:
            matched.append(i)
        if np.array_equal(feature[i], np.array([-1,-1,-1,-1])) is False and np.array_equal(landmarks[i], np.array([0,0,0,0])) is True:
            new.append(i)
 
    return new, matched

#Skew symetric matrix creator
def hatMap(A):
    return np.array([[0,-A[5],A[4],A[0]],[A[5],0,-A[3],A[1]],[-A[4],A[3],0,A[2]],[0,0,0,0]])

def curlHat(A):
    return np.array([[0,-A[5],A[4],0,-A[2],A[1]],
                     [A[5],0,-A[3],A[2],0,-A[0]],
                     [-A[4],A[3],0,-A[1],A[0],0],
                     [0,0,0,0,-A[5],A[4]],
                     [0,0,0,A[5],0,-A[3]],
                     [0,0,0,-A[4],A[3],0]])

#Extracts pose information 
def veeMap(A):
    return np.array([A[0][3],A[1][3],A[2][3],A[2][1],A[0][2],A[1][0]])

#Projection function used in transforming to pixel coordinates
def project(A):
    return A/A[2]

#Jacobian
def deltaProjection(A):
    return np.array([[1,0,-A[0]/A[2], 0],
                     [0, 1, -A[1]/A[2], 0],
                     [0,0,0,0],
                     [0,0,-A[3]/A[2],1]])*(1/A[2])

#Special operator for pose update
def identitySkew(A):
    return np.array([[1,0,0,0,A[2],-A[1]],
                     [0,1,0,-A[2],0,A[0]],
                     [0,0,1,A[1],-A[0],0],
                     [0,0,0,0,0,0]])
    

#if __name__ == '__main__':
SEPoseMatrix = np.array([expm(hatMap([0,0,0,0,0,0]))])
sePoseMatrix = np.array([[0,0,0,0,0,0]])
zMatrix = np.array([[0,0,0]])


filename = "./data/0022.npz"
time,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

fsu,fsv,cu,cv = K[0][0],K[1][1],K[0][2],K[1][2]
M = np.array([[fsu,0,cu,0],[0,fsv,cv,0],[fsu,0,cu,-fsu*b],[0,fsv,cv,0]])
coor= []

robot = KalmanSlamRobot()
robot.initLandmarks(features[:,:,0].T, K, b, cam_T_imu)
T = cam_T_imu
print("Total Time: ", str(int(len(time[0])*4/4)))
for t in range(1,int(len(time[0])*4/4)):
    vel = linear_velocity[:,t]
    rot = rotational_velocity[:,t]
    feature = features[:,:,t].T
    delt = time[0][t]-time[0][t-1]
    
    robot.predictPose(vel, rot, delt, T)
    robot.updatePose(feature, T, M)
    robot.updateLandmarks(feature, T, M, b)

#Displays grid map
if True:
    robot.grid[robot.landmarks[:,0].astype(int) + int(SHAPE[0]/2),robot.landmarks[:,1].astype(int) + int(SHAPE[1]/2)] = 1
    x = np.around(robot.sepose[:,0]).astype(int)
    y = np.around(robot.sepose[:,1]).astype(int)
    robot.grid[x,y] = -0.5
    imshow(robot.grid)
