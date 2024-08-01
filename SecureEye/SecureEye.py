#This is going to be for abbu's store to capture people at night


#first pip install these real quick

#cv2 because we need to process image and videos
import cv2

#torch because we need their object detection and classification
import torch

# numpy because we need it to process our image data and also deal with the image/video pixels on the screen and other math stuff when it comes to the camera
import numpy as np

#pygame because we need to the graphic part of it and also for the sound
import pygame

#next since we are doing an alarm sound we need to have the path to our alarm
sound = "alarm.wav"

#after the sound we need to be able to handle the sound so initiliaze pygame
pygame.init()
pygame.mixer.music.load(sound)


#here is loading the model for our detection
#the model we are loading is an object detection model that can detect and already name images and videos its a pretrained model instead of doing it ourselves
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#here we are going to opening the videos of the theives to test it, but we can also set it to the camera which we will hook up for nightshift
#cap = cv2.VideoCapture("thief_video.mp4")

#if you want to do a llive footage then do this:
cap = cv2.VideoCapture(0)


#here since we already have a trained model we just have to tell YOLOv5 which objects we are looking
target = ["car", "bus", "truck", "person"]


#we have to add count because we need to make sure it runs smoooth
count = 0

#and then when someone does enter danger zone we need pictures of them
pictures = 3

#so now this is going to be the points of interest so it will store in here 
pts = []


#now we need to type a function to draw out the inital polygon
#event is for when we click right or left
#x,y is the coordinates from clicking
def draw_polygon(event, x, y, flags, param):
    #here we need to make poitns a global variable because we can modify it
    global pts
    #now we are going to handle the mouse events
    #basic explanation if left button is clicked print the informatino then add to the recorded points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        pts.append([x,y])
    
    #what if we make a mistake and want to clear it?
    #press the right button
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts=[]

#now checking for the points inside the polygon
#point is a points
#polygon is defined by points and 0 and 1 is the points to test, and then false is to show if its inside or out
def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    #so if it is inside the polygon then its trye 
    if result == 1:
        return True
    #so if the point is outside or the edge then false we only want inside
    else:
        return False

#now we will create a window named Camera
cv2.namedWindow("Video")

#here we will not handle when we draw into our camera
#we use the callback function because this allows you to interact with the window
cv2.setMouseCallback("Video", draw_polygon)

#so this is for our pictures now
def preprocess(img):
    #this is gets the dimension of the image
    height, width = img.shape[:2]
    #here we need to get the ratio of the image which we then will use to resize
    ratio = height/width
    #resive with a 640 width, and then we will make int of 640 x the ratio for the new heigh
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

#here we use a while true because we want to be the ones to end it
while True:
    #ret = true or false to show if the frame was read
    #frame = actual video frame
    #.read goes frame by frame
    ret, frame = cap.read()
    #if ret is false and we dont read a frame
    if not ret:
        print("Failed. Try again")
        break
    #now creates a copy of the current frame
    frame_detected = frame.copy()
    #now we resize the frame so we can use the object detection
    frame = preprocess(frame)
    #now we run the code and store it in results
    results = model(frame)
    #now we will loop through each deteched frame
    #results.pandas().xyxy[0].iterrows(): Converts the detection results into a pandas DataFrame format and iterates through each row (each detected object).
    for index, row in results.pandas().xyxy[0].iterrows():
        #variables to store teh center of the detected object not needed but will pinpoint what we looking at
        center_x = None
        center_y = None 
        #here we are checking through the targets name and then our name will store the names of the detected object
        if row["name"] in target: 
            name = str(row["name"])
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            #now  we calculates the center of our polygon box
            center_x = int((x1+x2) / 2)
            center_y = int(y1 + y2 / 2)

            """
            The cv2.rectangle line draws a yellow rectangle around the detected object.
            The cv2.putText line places the name of the detected object near the top-left corner of the rectangle.
            The cv2.circle line marks the center of the detected object's bounding box with a red filled circle.
            """

            #now drawing bound box
            #cv2.rectang;e = function to draw recetangle on an image
            #frame = like the imge on which the rectangle will be drawn on so like which frame
            # the first coordinate pair = the top left corner of hte rectangle
            #the second coordinate pair is the bottom right corner of the rectable
            # the 255 255 0 is the BGR color formate so it will represent yellow
            # the 3 is just the thicnkness of the broder
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 0), 3)

            #now putting the text
            #cv2.puttext = draw text on the image
            #frame is just where it will be drawn on
            #name is the string format of the detected object thats from the trained model
            #first coordinate pair is the bottom left corner of the string in the image
            #the font hersehy is teh font
            # 1 is teh scale factor
            #that 255 255 0 is yellow
            #the 2 is thickness
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            #drawing the center point to specify when passed over the rectangle
            #cv2.cricle is just getting the circular dot on the object
            # frame is yk the frame 
            # so now the two center coordinate is getting teh mid point from earlier
            #5 is the radius of the circle
            # 0, 0 255 is the bgr format thsi way represents red
            # -1 indicates that the circle should be filled. If a positive value is used, it will represent the thickness of the circle's border.
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    

        #here we are going to start drawing the polygon
        #first we will check if there are 4 points on the list because we will make our polygon only after 4 points
        if len(pts) >= 4:
            #this creates a copy of the current frame
            #frame.copy is used to draw the polygon without altering the frame
            frame_copy = frame.copy()

            #now drawing the FILLEDDDD polygon
            #framecopy is the frame we are taking to draw on
            #np.arrary([points]) converts the points we clicked into a format that cv2.fillpoly can use
            #(0,255,0) this is teh bgr format green
            cv2.fillPoly(frame_copy, [np.array(pts)], (0, 255, 0))
            
            #now we are going to combine the filled polygon and the frame
            #cv2.addweighted is a function that blends two images together
            #frame_copy is the image with the filled polygon
            #.1 is the weight of frame_copy in the blending process it deteremines how much influence frame_copy has on the final image
            #0.9 is the weight of the original frame in the blending process. It determines how much influence the original frame has on the final image.
            #0 is the scalar added to each sum, which is 0 in this case.
            #The resulting image (frame) is a blend of the original frame and the frame with the filled polygon, with the polygon being slightly transparent.
            frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)

            #okay so now here we are going to be checking if someone is actualyl inside the polygon so the sound can play
            #we use is not none to check if the center coordinate of the detected objects are valid
            if center_x is not None and center_y is not None:
                #right here we are checking if the center of the person is in the polygon and its also named person
                if inside_polygon((center_x, center_y), np.array([pts])) and name == "person":
                    #this creates a blank mask of the same size as the detected frame
                    mask = np.zeros_like(frame_detected)
                    #this part defines the rectangle around the person
                    points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
                    #Reshapes the points array to the required shape for cv2.fillPoly.
                    points = points.reshape((-1, 1, 2))
                    #here it fills the mask with white color in teh detected rectangle
                    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
                    #so now we are going to apply the mask to the detected frame
                    frame_detected = cv2.bitwise_and(frame_detected, mask)

                    #now we are going to save the image
                    #check if teh count of saved photos is less than the required number of photoa
                    if count < pictures:
                        #Save the new frame picture as an image file
                        cv2.imwrite("Detected photos" + str(count) + ".jpg", frame_detected)

                    #now we are going to be playing the alarm
                    #first check if the alarm is playing
                    if not pygame.mixer.music.get_busy():
                        #now play the alarm sound
                        pygame.mixer.music.play()
                        #sets a flag to indicate the alarm is palying
                        alarm_playing = True

                    #now here we are going to change the screen and put text that shows that we detected someone
                    #for this first one we are going to write whatever label we decide to put
                    # 1, (0, 0, 255), 2) this means 1 is the font scale, then color for red, then 2 pixel thicc
                    cv2.putText(frame, "Intruder", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #now for this one we will write person detected at top-left corner of the frame
                    #(20,20) is the coorindates of where we want it
                    cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #now we draw teh red rectangle around the person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    #now increment the saved photos
                    count += 1

    #now this shows the current video in a window titled video
    cv2.imshow("Video", frame)
    #now this is how you get out of the video 
    #This function waits for a key event for a specified amount of time (1 millisecond in this case). It returns the ASCII value of the key pressed.
    #& 0xFF: This bitwise AND operation ensures compatibility across different systems by masking out all but the lowest 8 bits of the waitKey result.
    #ord('q'): This function returns the ASCII value of the character 'q'.


    #basically checks if the 'q' key was pressed. If it was, the loop breaks, stopping the video processing.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#now we close out.
# Release the video capture object to free up resources
cap.release()
#close it
cv2.destroyAllWindows()


























