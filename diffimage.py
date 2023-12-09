# Import packages
import cv2
import imutils
import numpy as np

# read image
img = cv2.imread("images/cow1.jpg")
img = cv2.resize (img, (600,360) )
height, width, channel = img.shape
print('original image shape:', height, width, channel)

# get blob from image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
print('blob shape:', blob.shape)

# read coco object names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print('number of classes =', len(classes))

# load pre-trained yolo model from configuration and weight files
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# set output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print('output layers:', output_layers)

# detect objects
net.setInput(blob)
outs = net.forward(output_layers)

# get bounding boxes and confidence socres
class_ids = []
confidence_scores = []
boxes = []

for out in outs: # for each detected object

    for detection in out: # for each bounding box

        scores = detection[5:] # scores (confidence) for all classes
        class_id = np.argmax(scores) # class id with the maximum score (confidence)
        confidence = scores[class_id] # the maximum score

        if confidence > 0.5:
            # bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)

print('number of dectected objects =', len(boxes))

# non maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
print('number of final objects =', len(indices))

# draw bounding boxes with labels on image
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(f'class {label} detected at {x}, {y}, {w}, {h}')
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, label, (x, y + 10), font, 1, color, 1)

cv2.imshow('Objects', img)

# Load the two images (city)
img1 = cv2.imread( "images/cow1.jpg")
img1 = cv2.resize (img1, (600,360) )
img2 = cv2.imread ("images/cow2.jpg")
img2 = cv2.resize (img2, (600,360) )

# Grayscale
gray1 = cv2.cvtColor (img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor (img2, cv2.COLOR_BGR2GRAY)

# Find the difference between the two images using absdiff
diff = cv2.absdiff(gray1, gray2)
cv2.imshow("diff(img1, img2)", diff)

# Apply threshold
thresh = cv2. threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2. THRESH_OTSU)[1]
cv2.imshow ("Threshold", thresh)

# Dilation
kernel = np.ones ((5,5), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=2)
cv2.imshow ("Dilation", dilate)

# Find contours
contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Loop over each contour
for contour in contours:
    if cv2. contourArea (contour) > 100:
        # Calculate bounding box
        x, y, w, h= cv2. boundingRect (contour)
        print(f'detected at {x}, {y}, {w}, {h}')
        # Draw rectangle - bounding box
        cv2.rectangle(img1, (x,y), (x+w, y+h) , (0,0,255) , 2)
        roi_img1 = img[y:y + h, x:x + w]
        cv2.rectangle(img2, (x,y) , (x+w, y+h), (0,0,255), 2)

cv2.imwrite("images/roi_img1.jpg", roi_img1)
cv2.imshow("images/roi_img1.jpg", roi_img1)

# Show final images with differences
x = np. zeros ((360, 10,3), np.uint8)
result = np.hstack((img1, x, img2))
cv2.imshow("Differences", result)

cv2.waitKey(0)
cv2.destroyAllWindows()