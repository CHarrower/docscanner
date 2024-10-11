import cv2
import numpy as np
import os

# Output folder path
output_folder = "** add your own file path **"
os.makedirs(output_folder, exist_ok=True)

# Function to reorder points for perspective transform
def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]  # Top-left point
    new_points[3] = points[np.argmax(add)]  # Bottom-right point
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]  # Top-right point
    new_points[2] = points[np.argmax(diff)]  # Bottom-left point
    return new_points

# Function to get the biggest contour
def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:  # Adjust this threshold as needed
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:  # Check for 4-point contour
                biggest = approx
                max_area = area
    return biggest

# Function to warp the perspective of the image
def warp_image(img, points, width, height):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (width, height))
    return img_warp

# Function to sharpen the image
def sharpen_image(img):
    # Creating a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

# Function to crop the image
def crop_image(img):
    height, width = img.shape[:2]
    # Crop a small border to remove any potential edges
    crop_margin = 10 
    return img[crop_margin:height - crop_margin, crop_margin:width - crop_margin]

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set camera width
cap.set(4, 720)   # Set camera height
cap.set(cv2.CAP_PROP_FOCUS, 0)  # Set focus if possible (0 for manual focus)

count = 0
scan_triggered = False
heightImg, widthImg = 720, 640

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab the frame")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 150)
    imgDilated = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=2)
    imgThresh = cv2.erode(imgDilated, np.ones((5, 5)), iterations=1)

    # Get the largest contour (document)
    biggest_contour = get_contours(imgThresh)

    if biggest_contour.size != 0:
        # Draw the contour for visualization
        cv2.drawContours(img, biggest_contour, -1, (0, 255, 0), 10)
        # Warp the image to a top-down view
        imgWarp = warp_image(img, biggest_contour, widthImg, heightImg)

        # Sharpen the warped image
        imgWarp = sharpen_image(imgWarp)

        # Crop the image to remove edges
        imgWarp = crop_image(imgWarp)

        # Show the warped (scanned-like) image
        cv2.imshow("Warped Image", imgWarp)

        # If scan triggered, save the warped image
        if scan_triggered:
            output_file_path = os.path.join(output_folder, f"scanned_image_{count}.jpg")
            cv2.imwrite(output_file_path, imgWarp)
            print(f"Scanned image saved to {output_file_path}")
            scan_triggered = False
            count += 1

    # Show the live feed with contour detection
    cv2.imshow("Live Feed", img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit with 'q'
        break
    elif key == ord(' '):  # Press spacebar to scan
        scan_triggered = True

cap.release()
cv2.destroyAllWindows()
