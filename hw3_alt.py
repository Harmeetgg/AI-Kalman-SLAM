import numpy as np
from utils import load_data
from skimage.io import imshow
from scipy.linalg import expm, logm, inv
from numpy.linalg import multi_dot

#Shape of grid
SHAPE = (10000,10000)

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
        
        #Track small se space of robot pose, initializing at middle of map and no rotations
        self.sepose = np.array([[SHAPE[0]/2,SHAPE[1]/2,10,0,0,0]])
        
        #Initialize covariance matrix of pose
        self.invSEcov  = np.array([np.identity(6)*0.1])
        
        #landmark and covariance array to be inistantiated 
        self.landmarks = None
        self.landCov = None
        self.featlength = None
    
    #Part1: Predict IMU Pose Only
    def predictPose(self, lin, rot, delt, T):
        
        #Pose in se3 space
        sePose = np.append(lin,rot)
        
        #Pose and covariance in SE3 space
        invSEPose = expm(-1*hatMap(sePose)*delt)
        invSEcov = expm(-1*curlHat(sePose)*delt)
        
        #Change the pose mean based on new IMU data
        self.invSEPose = np.append(self.invSEPose,[invSEPose.dot(self.invSEPose[-1])], axis = 0)

        
        #Change the pose covariance based on IMU and W variable and append as new element
        self.invSEcov = np.append(self.invSEcov, [invSEcov.dot(self.invSEcov[-1].dot(invSEcov.T)) + W], axis = 0)
        
        #Keep track of se3 version of pose to update grid map easily
        self.sepose = np.append(self.sepose,[veeMap(inv(self.invSEPose[-1]))] + self.sepose[0], axis = 0)
    
    #Part 2: Update landmarks based on IMU and EKF
    def predictObservations(self, T, M):
        
        predLandmarks = []
        for landmark in self.landmarks.reshape((int(self.featlength/3),3)):
            if np.array_equal(landmark, np.array([0,0,0])) is False:
                predLandmarks.append(M.dot(project(T.dot(self.invSEPose[-1].dot(landmark)))))
            else:
                predLandmarks.append([0,0,0])
        
        
        return np.array(predLandmarks)
        
    def mapUpdate(self, feature, T, M, b):
        zpred = self.predictObservations(T,M)
        U = self.invSEPose[-1]
        new, matched = matchZDisparity(feature, zpred)
        
        
        
        
        
        
        for i in matched:
            proj = deltaProjection(multi_dot([T,U,self.landmarks[i]])).T
            H = multi_dot([M,proj,T,U,P.T])
            sig = self.landCov[i]
            K = multi_dot([sig,H.T,inv(multi_dot([H,sig,H.T]) + np.identity(4)*V)])
            
            self.landmarks[i] = self.landmarks[i] + np.append(K.dot(feature[i]-zpred[i]),0)
            self.landCov[i] = (np.identity(3) - K.dot(H)).dot(self.landCov[i])
        if len(new) != 0:
            
            self.addLandmarks(feature, new, T, M, b)
        
        
    def addLandmarks(self, feature, new, T, M, b):
        fsu,cu,cv = M[0][0],M[0][2],M[1][2]
        for i in new:
            landmark = feature[i]
            disparity = landmark[0] - landmark[2]
            x = ((landmark[0]-cu)*b)/(disparity)
            y = ((landmark[1]-cv)*b)/(disparity)
            z = b*fsu/disparity
        self.landmarks[i*3:i*3+3] = inv(self.invSEPose[-1]).dot(inv(T).dot(np.array([x,y,z,1])))[:-1]
        
        
        
    def update(self, feature, T, M):
        zpred = self.predictObservations(T,M)
        H = self.H(T,M)
        new, matched = matchZDisparity(feature, zpred.T)
        
        K = np.array([self.SEcov[-1].dot(h.T.dot(h.dot(self.SEcov[-1].dot(h.T))+np.identity(4)*V)) for h in H])
        
        return K


    def H(self, T, M):
        
        inner = self.invSEPose[-1].dot(self.landmarks.T)
        return np.array([M.dot(deltaProjection(T.dot(A.T)).dot(T.dot(identitySkew(A)))) for A in inner.T])
    

    
    def initLandmarks(self, feature, K, b, T):
        fsu,cu,cv = K[0][0],K[0][2],K[1][2]
        m = []
        for landmark in feature:
            if np.array_equal(landmark, np.array([-1,-1,-1,-1])) is False:
                disparity = landmark[0] - landmark[2]
                x = ((landmark[0]-cu)*b)/(disparity)
                y = ((landmark[1]-cv)*b)/(disparity)
                z = b*fsu/disparity
            
                m.extend(inv(self.invSEPose[-1]).dot(inv(T).dot(np.array([x,y,z,1])))[:-1])
            else:
                m.extend([0,0,0])
                
        m = np.array(m)
        
        self.landmarks = np.array(m)
        self.landCov = np.identity(len(feature)*3)*V
        self.featlength = len(feature)
            

#Find new and matched arrays
def matchZDisparity(feature, zpred):
    assert len(feature) == len(zpred)
    new = []
    matched = []
    
    for i in range(len(feature)):
        if np.array_equal(feature[i], np.array([-1,-1,-1,-1])) is False and np.array_equal(zpred[i], np.array([0,0,0])) is False:
            matched.append(i)
        if np.array_equal(feature[i], np.array([-1,-1,-1,-1])) is False and np.array_equal(zpred[i], np.array([0,0,0])) is True:
            new.append(i)
 
    return new, matched

#Closed form function for exponential map using rodrigues formula
def expmap(A):
    
    dim = len(A)
    mag = np.linalg.norm(A)
    R = np.identity(dim) + (np.sin(mag)/mag)*A + ((1-np.cos(mag))/(mag**2))*A.dot(A)
    
    return R

def hatMap(A):
    return np.array([[0,-A[5],A[4],A[0]],[A[5],0,-A[3],A[1]],[-A[4],A[3],0,A[2]],[0,0,0,0]])

def curlHat(A):
    return np.array([[0,-A[5],A[4],0,-A[2],A[1]],[A[5],0,-A[3],A[2],0,-A[0]],[-A[4],A[3],0,-A[1],A[0],0],
                     [0,0,0,0,-A[5],A[4]],[0,0,0,A[5],0,-A[3]],[0,0,0,-A[4],A[3],0]])
    
def veeMap(A):
    return np.array([A[0][3],A[1][3],A[2][3],A[2][1],A[0][2],A[1][0]])
    
def project(A):
    return A/A[2]

def deltaProjection(A):
    return np.array([[1,0,-A[0]/A[2], 0],[0, 1, -A[1]/A[2], 0],[0,0,0,0],[0,0,-A[3]/A[2],1]])*(1/A[2])

def identitySkew(A):
    return np.array([[1,0,0,0,A[2],-A[1]],[0,1,0,-A[2],0,A[0]],[0,0,1,A[1],A[0],0],[0,0,0,0,0,0]])
    

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
#robot.mapUpdate(features[:,:,0].T, cam_T_imu, M, b)

for t in range(1,2):#len(time[0])):
    vel = linear_velocity[:,t]
    rot = rotational_velocity[:,t]
    feature = features[:,:,t].T
    delt = time[0][t]-time[0][t-1]

    robot.predictPose(vel, rot, delt, cam_T_imu)
    #robot.mapUpdate(feature, cam_T_imu, M, b)


    
robot.grid[robot.landmarks[:,0].astype(int) + int(SHAPE[0]/2),robot.landmarks[:,1].astype(int) + int(SHAPE[1]/2)] = 1
x = np.around(robot.sepose[:,0]).astype(int)
y = np.around(robot.sepose[:,1]).astype(int)
robot.grid[x,y] = -0.5
imshow(robot.grid)






    # def altupdatePose(self, feature, T, M):
    #     zpred = self.predictObservations(T,M)
    #     new, matched = matchZ(feature, zpred)
    #     U = self.invSEPose[-1]
    #     sig = self.invSEcov[-1]
        
    #     mean = []
    #     cov = []
    #     for i in matched:
            
    #         proj = deltaProjection(multi_dot([T,U,self.landmarks[i]]))
    #         H = multi_dot([M,proj,T,identitySkew(U.dot(self.landmarks[i]))])
    #         K = multi_dot([sig,H.T,inv(multi_dot([H,sig,H.T]) + np.identity(4)*V)])
    #         corr = (feature[i] - zpred[i])
    #         mean.append(expm(hatMap(K.dot(corr))).dot(self.invSEPose[-1]))
    #         cov.append((np.identity(6) - K.dot(H)).dot(sig))

    #     mean = np.array(mean)
    #     #print(mean)
    #     #print(U)
    #     cov = np.array(cov)
        
    #     meansum = np.sum(np.sum(mean, axis = 1), axis= 1)
    #     idx = np.where(meansum == np.amin(meansum))[0][0]
        
    #     self.invSEPose = np.append(self.invSEPose, [mean[idx]], axis = 0)
    #     self.invSEcov  = np.append(self.invSEcov,  [cov[idx]], axis = 0)
        
    #     #Keep track of se3 version of pose to update grid map easily
    #     self.sepose = np.append(self.sepose,[veeMap(inv(self.invSEPose[-1]))] + self.sepose[0], axis = 0)
        
    #     return mean








        # K = []
        # for i in range(len(matched)):
        #     try:
        #         K.append(multi_dot([sig,H[i].T,inv(multi_dot([H[i],sig,H[i].T]) + np.identity(4)*V)]).T)
        #     except:
        #         print("Error Raised")
        #         print(proj[i])
        #         print(proj[i-1])
        #         print(H[i])
        #         print(H[i-1])
        #         print(sig)
        #         print(multi_dot([H[i],sig,H[i].T]))    
        # K = np.array(K)

















    # feature = features[:,:,t].T
    # filtfeature = feature[feature>0]
    # filtfeature = filtfeature.reshape((int(len(filtfeature)/4),4)).T
    # disparity = filtfeature[0] - filtfeature[2]
    # depth = fsu*b/disparity
    # x = ((filtfeature[0]-cu)*b)/(disparity)
    # y = ((filtfeature[1]-cv)*b)/(disparity)
    
    #print(filtfeature)
    
    # normcoor = M.dot(filtfeature)#*depth
    # normcoor = normcoor/normcoor[-1]*depth
    # normcoor = normcoor[:-1]
    
    
    #normcoor[:2] = normcoor[:2]/normcoor[2]
    #normcoor[2] = normcoor[2]/normcoor[2]
    
    
    # vel = linear_velocity[:,t]
    # rot = rotational_velocity[:,t]
    
    
    # sePose = np.append(rot,vel)
    # SEPose = expm(hatMap(sePose)*(time[0][t]-time[0][t-1]))
    
    
    # SEPoseMatrix = np.append(SEPoseMatrix,[SEPoseMatrix[-1].dot(SEPose)], axis = 0)
    # sePoseMatrix = np.append(sePoseMatrix,[veeMap(logm(SEPoseMatrix[-1]))], axis = 0)
    
    # observation = M.dot(project(SEPose.dot(np.linalg.inv(cam_T_imu).dot(normcoor))))
    # zMatrix = np.append(zMatrix, observation)
    
    
    
# print(normcoor)
# #print(filtfeature.T[0])
# grid = np.zeros((2000,2000))

# x, y = np.real(sePoseMatrix)[:,3].astype(int)+1000, np.real(sePoseMatrix)[:,4].astype(int)+1000

# grid[x,y] = 1

#imshow(grid)
#print(logm(SEPoseMatrix[-1]))


    #print(t)

    #Integrate velocities to get absolute positions and orientations over time 
    
    
	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
