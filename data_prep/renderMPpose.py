import cv2
import json
import numpy as np
import math
import mediapipe as mp


pose_colors = [[255,     0,    85], \
		[255,     0,     0], \
		[255,    85,     0], \
		[255,   170,     0], \
		[255,   255,     0], \
		[170,   255,     0], \
		[85,   255,     0], \
		[0,   255,     0], \
		[255,     0,     0], \
		[0,   255,    85], \
		[0,   255,   170], \
		[0,   255,   255], \
		[0,   170,   255], \
		[0,    85,   255], \
		[0,     0,   255], \
		[255,     0,   170], \
		[170,     0,   255], \
		[255,     0,   255], \
		[85,     0,   255], \
		[0,     0,   255], \
		[0,     0,   255], \
		[0,     0,   255], \
		[0,   255,   255], \
		[0,   255,   255], \
		[0,   255,   255]]
    
hand_colors = [[230, 53, 40], [231,115,64], [233, 136, 31], [213,160,13],[217, 200, 19], \
    [170, 210, 35], [139, 228, 48], [83, 214, 45], [77, 192, 46], \
    [83, 213, 133], [82, 223, 190], [80, 184, 197], [78, 140, 189], \
    [86, 112, 208], [83, 73, 217], [123,46,183], [189, 102,255], \
    [218, 83, 232], [229, 65, 189], [236, 61, 141], [255, 102, 145]]

faceSeq = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10,11], [11,12], [12,13], [13,14], [14,15], [15,16], \
    [17,18], [18,19], [19,20], [20,21], [22,23], [23,24], [24,25], [25,26], \
    [27,28], [28,29], [29,30], [31,32], [32,33], [33,34], [34,35], \
    [36,37], [37,38], [38,39], [39,40], [40,41], [41,36], [42,43], [43,44], [44,45], [45,46], [46,47], [47,42], \
    [48,49], [49,50], [50,51], [51,52], [52,53], [53,54], [54,55], [55,56], [56,57], [57,58], [58,59], [59,48], [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,60]]

handSeq = [[0,1], [1,2], [2,3], [3,4], \
    [0,5], [5,6], [6,7], [7,8], \
    [0,9], [9,10], [10,11], [11,12], \
    [0,13], [13,14], [14,15], [15,16], \
    [0,17], [17,18], [18,19], [19,20], \
    [5,9], [9,13], [13,17]]


def readkeypointsfile_json(myfile):
	import json
	f = open(myfile, 'r')
	json_dict = json.load(f)
	people = json_dict['people']
	posepts =[]
	facepts = []
	r_handpts = []
	l_handpts = []
	for p in people:
		posepts += p['pose_keypoints_2d']
		facepts += p['face_keypoints_2d']
		r_handpts += p['hand_right_keypoints_2d']
		l_handpts += p['hand_left_keypoints_2d']

	return posepts, facepts, r_handpts, l_handpts

def display_hand_skleton(frame, r_handpts, l_handpts):
        
                
    for k in range(len(handSeq)):
        firstlimb_ind = handSeq[k][0]
        secondlimb_ind = handSeq[k][1]
        cv2.line(frame, (int(r_handpts[firstlimb_ind, 0]), int(r_handpts[firstlimb_ind, 1])), (int(r_handpts[secondlimb_ind, 0]), int(r_handpts[secondlimb_ind, 1])), (255,255,255), 4)
        cv2.line(frame, (int(l_handpts[firstlimb_ind, 0]), int(l_handpts[firstlimb_ind, 1])), (int(l_handpts[secondlimb_ind, 0]), int(l_handpts[secondlimb_ind, 1])), (255,255,255), 4)

    for p in range(r_handpts.shape[0]):
        cv2.circle(frame, (int(r_handpts[p,0]), int(r_handpts[p,1])), 4, (255, 0, 255), -1)
        cv2.circle(frame, (int(l_handpts[p,0]), int(l_handpts[p,1])), 4, (255, 0, 255), -1)   
            
    return True
            
def resize_scale(frame, myshape = (512, 1024, 3)):
    curshape = frame.shape
    if curshape == myshape:
        scale = 1
        translate = (0.0, 0.0)
        return scale, translate

    x_mult = myshape[0] / float(curshape[0])
    y_mult = myshape[1] / float(curshape[1])

    if x_mult == y_mult:
        scale = x_mult
        translate = (0.0, 0.0)
    elif y_mult > x_mult:
        y_new = x_mult * float(curshape[1])
        translate_y = (myshape[1] - y_new) / 2.0
        scale = x_mult
        translate = (translate_y, 0.0)
    elif x_mult > y_mult:
        x_new = y_mult * float(curshape[0])
        translate_x = (myshape[0] - x_new) / 2.0
        scale = y_mult
        translate = (0.0, translate_x)
        
    # M = np.float32([[scale,0,translate[0]],[0,scale,translate[1]]])
    # output_image = cv2.warpAffine(frame,M,(myshape[1],myshape[0]))
    return scale, translate

def fix_image(scale, translate, frame, myshape = (512, 1024, 3)):
    M = np.float32([[scale,0,translate[0]],[0,scale,translate[1]]])
    output_image = cv2.warpAffine(frame,M,(myshape[1],myshape[0]))
    return output_image

def fix_scale_coords(points, scale, translate):
    points = np.array(points)
    points[0::3] = scale * points[0::3] + translate[0]
    points[1::3] = scale * points[1::3] + translate[1]
    return list(points)

def rescale_points(width, height, output):
    output[:,0] = output[:,0]*width
    output[:,1] = output[:,1]*height
    return output

def display_single_hand_skleton(frame, handpts):
                        
    for k in range(len(handSeq)):
        firstlimb_ind = handSeq[k][0]
        secondlimb_ind = handSeq[k][1]
        cv2.line(frame, (int(handpts[firstlimb_ind, 0]), int(handpts[firstlimb_ind, 1])), (int(handpts[secondlimb_ind, 0]), int(handpts[secondlimb_ind, 1])), (255,255,255), 4)

    for p in range(handpts.shape[0]):
        cv2.circle(frame, (int(handpts[p,0]), int(handpts[p,1])), 4, (255, 0, 255), -1)
            
    return True

def make_bbox(handpts, size):
    x_mid = int(np.average(handpts[:,0]))
    y_mid = int(np.average(handpts[:,1]))
    
    if x_mid - size/2 > 0:
        sx = x_mid-size/2
    else:
        sx = 0
        
    if y_mid - size/2 > 0:
        sy = y_mid-size/2
    else:
        sy = 0
        
    return int(sx), int(sx+size), int(sy), int(sy+size)

def restructure_points(handpts, sx, sy):
    hpts = handpts
    hpts[:,0] = hpts[:,0] - sx
    hpts[:,1] = hpts[:,1] - sy 
    return hpts.astype(int)


def assert_bbox(handpts, size):
    x_mid = int(np.average(handpts[:,0]))
    y_mid = int(np.average(handpts[:,1]))
    x_min = int(np.min(handpts[:,0]))
    y_min = int(np.min(handpts[:,1]))
    x_max = int(np.max(handpts[:,0]))
    y_max = int(np.max(handpts[:,1]))
    
    max_dis = max(abs(x_max-x_min), abs(y_max-y_min))
    
    if x_mid - max_dis/2 > 0:
        sx = x_mid - max_dis/2 - max_dis*0.25
    else:
        sx = 0
    
    if y_mid - max_dis/2 > 0:
        sy = y_mid - max_dis/2 - max_dis*0.25
    else:
        sy = 0
    
    return int(sx), int(sx + max_dis*1.5), int(sy), int(sy + max_dis*1.5)

mp_hands = mp.solutions.hands

def GetCoordForCurrentInstance(mp_output):
    hand_pts = np.zeros((21, 2))
    hand_pts[0, 0] = mp_output.landmark[mp_hands.HandLandmark.WRIST].x
    hand_pts[0, 1] = mp_output.landmark[mp_hands.HandLandmark.WRIST].y
    hand_pts[1, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_CMC].x
    hand_pts[1, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_CMC].y
    hand_pts[2, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_MCP].x
    hand_pts[2, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    hand_pts[3, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_IP].x
    hand_pts[3, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_IP].y
    hand_pts[4, 0] = mp_output.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    hand_pts[4, 1] = mp_output.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    hand_pts[5, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
    hand_pts[5, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    hand_pts[6, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    hand_pts[6, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    hand_pts[7, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
    hand_pts[7, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    hand_pts[8, 0] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    hand_pts[8, 1] = mp_output.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    hand_pts[9, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    hand_pts[9, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    hand_pts[10, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
    hand_pts[10, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    hand_pts[11, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
    hand_pts[11, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    hand_pts[12, 0] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
    hand_pts[12, 1] = mp_output.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    hand_pts[13, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
    hand_pts[13, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    hand_pts[14, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
    hand_pts[14, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    hand_pts[15, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
    hand_pts[15, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
    hand_pts[16, 0] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
    hand_pts[16, 1] = mp_output.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    hand_pts[17, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_MCP].x
    hand_pts[17, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_MCP].y
    hand_pts[18, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_PIP].x
    hand_pts[18, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    hand_pts[19, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_DIP].x
    hand_pts[19, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_DIP].y
    hand_pts[20, 0] = mp_output.landmark[mp_hands.HandLandmark.PINKY_TIP].x
    hand_pts[20, 1] = mp_output.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    return hand_pts
    