import numpy as np
import argparse
import os
import glob
from renderMPpose import *
import cv2
import mediapipe as mp
import pickle
from google.protobuf.json_format import MessageToDict
import signal

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--video_dir", type=str, help='video directory')
parser.add_argument("--display", help='display the output', action="store_true")
parser.add_argument("--output_pkl", type=str, help='output pkl file')

args = parser.parse_args()

complete_hand_landmarks = []

def keyboardInterruptHandler(signal, frame):
    if args.output_pkl:
        hand_dict = {'joints' : complete_hand_landmarks}
        outfile = open(args.output_pkl, 'wb')
        pickle.dump(hand_dict, outfile)
        outfile.close()
        print(f'Saved Output to: {args.output_pkl}')
    exit(0)
    
signal.signal(signal.SIGINT, keyboardInterruptHandler)


vids_path = os.path.join(args.video_dir, "*.mp4")
vids = glob.glob(vids_path)
vids.sort()
total_vids = len(vids)

print(f'Total Videos: {total_vids}')

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.9)

for v in range(len(vids)):
    print(f"Processing {v+1}/{len(vids)} : {vids[v]}")
    cap = cv2.VideoCapture(vids[v])
    while(cap.isOpened()):
        res, frame = cap.read()
        if res == True:
            scale_n, translate_n = resize_scale(frame)
            image = fix_image(scale_n, translate_n, frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            hand_side = [None] * 2
            if not results.multi_hand_landmarks:
                continue
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                hand_side[idx] = handedness_dict["classification"][0]["label"]
            if hand_side[0] == hand_side[1] or hand_side[0] == None or hand_side[1] == None:
                continue
            if results.multi_hand_landmarks:
                new_hand_landmarks = [None] * 2
                index = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    if hand_side[index] == "Left":
                        new_hand_landmarks[0] = hand_landmarks
                    elif hand_side[index] == "Right":
                        new_hand_landmarks[1] = hand_landmarks
                    index = index + 1
                complete_hand_landmarks.append(new_hand_landmarks)
        else:
            break
    cap.release()

        
hands.close()


if args.output_pkl:
    hand_dict = {'joints' : complete_hand_landmarks}
    outfile = open(args.output_pkl, 'wb')
    pickle.dump(hand_dict, outfile)
    outfile.close()
