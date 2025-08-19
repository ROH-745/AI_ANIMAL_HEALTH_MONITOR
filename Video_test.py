import cv2



video_path=r"C:\Users\Rohith\Desktop\AI_ANIMAL_MONITOR\videos\COW_VIDEO2.mp4"

width,height=640,480

cap=cv2.VideoCapture(video_path)




while True:
  ret,frame=cap.read()

  if not ret:
    break

  frame=cv2.resize(frame,(width,height))

  cv2.imshow("cow video",frame)


  if cv2.waitKey(25) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()
