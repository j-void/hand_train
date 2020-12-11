import joblib
import numpy as np
import mediapipe as mp
import cv2
from renderMPpose import *
import argparse
import signal
import pickle
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_pkl", type=str, help='input pkl')
parser.add_argument("--display", help='display the output', action="store_true")
parser.add_argument("--output_dir", type=str, help='output directory')
parser.add_argument("--hand", default="both", type=str, help='output data type: both / combine')

args = parser.parse_args()
output = joblib.load(args.input_pkl) 
handpts = output["joints"]

outdir = args.output_dir

if not os.path.exists(outdir):
    os.makedirs(outdir)

def save_output(pts, name):
    hand_dict = {'joints' : pts}
    outfile = open(name, 'wb')
    pickle.dump(hand_dict, outfile)
    outfile.close()
    print("Saving output: ", name)

print("Total hand ouput: ", len(handpts))

img = np.zeros((512, 1024, 3), dtype=np.uint8)
index = 100

lhpts_list = []
rhpts_list = []

if args.hand == 'combine':
    combine_list = []


def keyboardInterruptHandler(signal, frame):
    if args.hand == 'both':
        save_output(lhpts_list, os.path.join(outdir, "lhpts.pkl"))
        save_output(rhpts_list, os.path.join(outdir, "rhpts.pkl"))
    exit(0)
    
signal.signal(signal.SIGINT, keyboardInterruptHandler)

for i in range(len(handpts)):
    img = np.zeros((512, 1024, 3), dtype=np.uint8)
    img_l = np.zeros((128, 128, 3), dtype=np.uint8)
    img_r = np.zeros((128, 128, 3), dtype=np.uint8)
    
    if (handpts[i][0] == None or handpts[i][1] == None):
        continue
    
    left_h = rescale_points(1024, 512, GetCoordForCurrentInstance(handpts[i][0]))
    rigth_h = rescale_points(1024, 512, GetCoordForCurrentInstance(handpts[i][1]))
   
    display_hand_skleton(img, rigth_h, left_h)

    x_start_l, x_end_l, y_start_l, y_end_l = make_bbox(left_h, 128)
    left_h_new = restructure_points(left_h, x_start_l, y_start_l)
    cv2.rectangle(img, (x_start_l, y_start_l), (x_end_l, y_end_l), (255, 0, 0), 2)

    x_start, x_end, y_start, y_end = make_bbox(rigth_h, 128)
    rigth_h_new = restructure_points(rigth_h, x_start, y_start)
    cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

    img_l = np.zeros((128, 128, 3), dtype=np.uint8)
    img_r = np.zeros((128, 128, 3), dtype=np.uint8)
    
    lhpts_list.append(left_h_new)
    rhpts_list.append(rigth_h_new)

    display_single_hand_skleton(img_l, left_h_new)
    display_single_hand_skleton(img_r, rigth_h_new)
    
    print("Progress: ",int((i/(len(handpts)-1))*100),"%", end="\r")

    if args.display:
        cv2.imshow("Show", img)
        cv2.imshow("Left", img_l)
        cv2.imshow("Right", img_r)
        if cv2.waitKey(4) & 0xFF == ord('q'):
            break

if args.hand == 'both':
    save_output(lhpts_list, os.path.join(outdir, "lhpts.pkl"))
    save_output(rhpts_list, os.path.join(outdir, "rhpts.pkl"))
