import cv2, json

cap = cv2.VideoCapture("rtsp://admin:Admin123@172.168.0.6:554/h264/ch33/main/av_stream")
# ret, frame = cap.read()
while True:
    ret, frame = cap.read()
    cv2.imshow("frame",frame)
    # print("ret", type(ret), " frame:",type(frame))
    
    break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
