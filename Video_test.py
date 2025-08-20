import cv2
import mediapipe as mp
import pandas as pd

mp_pose=mp.solutions.pose
# pose=mp_pose.Pose(min_detection_confidence=0.3,min_tracking_confidence=0.3)
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing=mp.solutions.drawing_utils


video_path=r"C:\Users\Rohith\Desktop\AI_ANIMAL_MONITOR\videos\t2.mp4"
# video_path=r"C:\Users\Rohith\Desktop\AI_ANIMAL_MONITOR\videos\woman_walking.mp4"

width,height=640,480

cap=cv2.VideoCapture(video_path)


pose_data=[]

while True:
  ret,frame=cap.read()

  if not ret:
    break

  frame=cv2.resize(frame,(width,height))

  rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  results=pose.process(rgb_frame)

  if results.pose_landmarks:
    
    mp_drawing.draw_landmarks(
      frame,
      results.pose_landmarks,
      mp_pose.POSE_CONNECTIONS
      
    )
    
    landmarks=results.pose_landmarks.landmark

    row=[landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
    pose_data.append(row)

  cv2.imshow("cow video",frame)


  if cv2.waitKey(25) & 0xFF == 27:
    break

cap.release()


df = pd.DataFrame(pose_data)
df.to_csv('keypoints.csv', index=False)

print("Data saved to keypoints.csv!")

cv2.destroyAllWindows()
