import cv2

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread('test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)

# Display the image with the detected faces
cv2.imshow('Face Detection', img)

# Wait for a key press and close the window
cv2.waitKey(0)

# Save the image with rectangles drawn around faces
cv2.imwrite("face_detected.jpg", img)
