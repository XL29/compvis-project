from djitellopy import Tello
from threading import Thread
import cv2 as c

me = Tello()
me.connect()
print(me.get_battery())

me.streamon()
#me.set_video_direction(Tello.CAMERA_DOWNWARD)

def verticalRecording():
    me.set_video_direction(Tello.CAMERA_DOWNWARD)
    frame = me.get_frame_read().frame
    frame = c.resize(frame, (360, 240))
    c.imshow("Vetical Image", frame)
    c.waitKey(1)

def horizontalRecording():
    me.set_video_direction(Tello.CAMERA_FORWARD)
    frame = me.get_frame_read().frame
    frame = c.resize(frame, (360, 240))
    c.imshow("Horizontal Image", frame)
    c.waitKey(1)


verticalRecorder = Thread(target=verticalRecording)
horizontalRecorder = Thread(target=horizontalRecording)

while True:
    verticalRecording()
    horizontalRecording() 
    """ img = me.get_frame_read().frame
    #img = c.resize(img, (360, 240))
    c.imshow("Image", img)
    c.waitKey(1) """