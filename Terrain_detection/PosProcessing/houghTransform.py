import numpy as np
import cv2

# Load the .npy file
data = np.load('pts_dict_104700.npy', allow_pickle=True)

img = cv2.imread('img_104700.png')
print(data)
# Get image dimensions
height, width, channels = img.shape

# Create a binary image
binary_arr = np.zeros((height, width), dtype=np.uint8)

# Convert point list to integer and set white pixels
whitePixels = [(int(pts[0]), int(pts[1])) for pts in data]
for x, y in whitePixels:
    if 0 <= x < width and 0 <= y < height:
        binary_arr[y, x] = 255

cv2.imwrite('binary.png', binary_arr)

# Apply Hough Transform 
rho_threshold =  5 # Pixel step size
theta_threshold = np.pi / 180  # Angle resolution
min_votes = 3  # Minimum votes for a line
lines = cv2.HoughLines(binary_arr, rho_threshold, theta_threshold, min_votes)

# Define angle tolerance
angle_tolerance = 75  # Degrees
rho_tolerance = 30  # Distance tolerance

if lines is not None:
    valid_lines = []
    angles = []

    # Extract angles and convert to degrees
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        angles.append((rho, angle))  # Store (rho, angle) pairs

    # Check for parallel and perpendicular conditions
    for idx, (rho1, angle1) in enumerate(angles):  
        is_valid = False
        for rho2, angle2 in angles[idx+1:]:  #  Skip current element
            angle_diff = abs(angle1 - angle2)
            rho_diff = abs(rho1 - rho2)

            # Check for parallelism or perpendicularity with small rho difference
            if (angle_diff < angle_tolerance) or (abs(angle_diff - 90) < angle_tolerance):
                if not rho_diff < rho_tolerance:
                    is_valid = True
                    break  # Stop checking if we found a match

        if is_valid:
            valid_lines.append((rho1, np.radians(angle1)))  # Convert back to radians

    # Draw only valid lines
    for rho, theta in valid_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        print(rho, np.degrees(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for valid lines

    cv2.imwrite('filtered_houghlines.jpg', img)
    print(f"Filtered lines drawn: {len(valid_lines)}")
else:
    print("No lines detected.")
