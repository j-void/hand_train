import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
complete_hand_landmarks = []

hands = mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.9)
cap = cv2.VideoCapture("../../data/alphabets/alphabets.mp4")
frame_index = 0
while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
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
  #print(hand_side)
  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    new_hand_landmarks = [None] * 2
    index = 0
    for hand_landmarks in results.multi_hand_landmarks:
      if hand_side[index] == "Left":
        new_hand_landmarks[0] = hand_landmarks
      elif hand_side[index] == "Right":
        new_hand_landmarks[1] = hand_landmarks
      mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      index = index + 1
    complete_hand_landmarks.append(new_hand_landmarks)
  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()

hand_dict = {'joints' : complete_hand_landmarks}

outfile = open("2D_hands.pkl", 'wb')
pickle.dump(hand_dict, outfile)
outfile.close()

# image = cv2.flip(cv2.imread("../../data/alphabets/frames/abc_000000000032.png"), 1)
# # Convert the BGR image to RGB before processing.
# results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# # Print handedness and draw hand landmarks on the image.
# print('Handedness:', results.multi_handedness)
# for idx, hand_handedness in enumerate(results.multi_handedness):
#     handedness_dict = MessageToDict(hand_handedness)
#     print(handedness_dict["classification"][0]["label"])
# image_hight, image_width, _ = image.shape
# annotated_image = image.copy()
# for hand_landmarks in results.multi_hand_landmarks:
#   # print('hand_landmarks:', hand_landmarks)
#   # print(
#   #     f'Index finger tip coordinates: (',
#   #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#   #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
#   # )
#   mp_drawing.draw_landmarks(
#       annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
# cv2.imshow("Image",cv2.flip(annotated_image, 1))
# cv2.waitKey(0)
# hands.close()