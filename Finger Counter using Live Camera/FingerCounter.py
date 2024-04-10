import cv2 as cv
import HandTrackingModule as htm
import os
import time

# Camera configuration
cam_width = 640
cam_height = 480

cap = cv.VideoCapture(0)

cap.set(3, cam_width)
cap.set(4, cam_height)

# Imporint and storing fingers images
images = []

ImagesFolderPath = "Finger Images"
ImagesList = os.listdir(ImagesFolderPath)
for image in ImagesList:
    img = cv.imread(f"{ImagesFolderPath}/{image}", cv.IMREAD_UNCHANGED)

    images.append(img)

# For calculating the FPS
previous_time = 0

# Initializing the hand detector
detector = htm.HandDetector(detection_confidence=0.7)

# The IDs of the landmarks on finger tips
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    img = detector.findHands(img)

    landmarks = detector.find_position(img)

    # Initializing the overlay image (The finger image that appears)
    overlay_img = images[0]

    fingers = []

    if len(landmarks) != 0:
        # If right hand (Thumb_x < Pinky_x)
        if landmarks[4][1] < landmarks[20][1]:
            
            # If thumb is up
            if landmarks[4][1] < landmarks[3][1]:
                fingers.append(1)
            else: 
                fingers.append(0)
            
            # If other fingers are up
            for id in range(1, 5):
                if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        else: # Left hand

            # If thumb is up
            if landmarks[4][1] > landmarks[3][1]:
                fingers.append(1)
            else: 
                fingers.append(0)
            
            # If other fingers are up
            for id in range(1, 5):
                if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

    
        print(fingers)

        # Indexing the suitable image
        overlay_img = images[fingers.count(1)]
            
    # BGRA supports the alpha (transparent) channel
    img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

    # Setting the starting point of the position of the image
    x, y = cam_width - 200, 0 # The top left of the camera image
    
    # Assuming the overlay image and the section of the feed where it's being placed are the same size
    w, h = overlay_img.shape[1], overlay_img.shape[0] # Dimensions of the overlay image

    # Extract the alpha mask of the overlay image (Normalized then Reversed)
    alpha_overlay = overlay_img[:, :, 3] / 255.0 
    alpha_background = 1.0 - alpha_overlay

    # Loop through each channel (RGB)
    for c in range(0, 3):
        img[y:y+h, x:x+w, c] = (alpha_overlay * overlay_img[:, :, c] + alpha_background * img[y:y+h, x:x+w, c])

    # Calculating the FPS
    current_time = time.time()
    fps = round(1 / (current_time - previous_time))
    previous_time = current_time

    # Showing the FPS
    cv.putText(img, "FPS: " + str(fps), (10, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

    cv.imshow("Image", img)

    if cv.waitKey(1) == 27:
        break
