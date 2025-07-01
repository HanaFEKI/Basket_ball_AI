import cv2

#Open video
cap = cv2.VideoCapture('../data/videos/basket_game.mp4')
cap = cv2.VideoCapture('../../../TacTic/src/data/basket_game.mp4')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
    ret, img = cap.read()
    if ret:
        img_resized = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))) 
        cv2.imshow("S = Save Frame / ESC = Exit", img_resized)

cv2.namedWindow('S = Save Frame / ESC = Exit')

cv2.createTrackbar('frame', 'S = Save Frame / ESC = Exit', 0, length, onChange)

#Display first frame
onChange(0)

frames = [3]
current_frame = 0

while cap.isOpened():
    if cv2.getWindowProperty('S = Save Frame / ESC = Exit', cv2.WND_PROP_VISIBLE) < 1:
        break

    k = cv2.waitKey(10) & 0xff
    if k == 27: #ESC to close
        break
    elif k == ord('s'):  # Press S to save
        ret, img = cap.read()
        if ret:
            filename = f"../data/input_imgs/img_{frames[current_frame]}.png"
            cv2.imwrite(filename, img)
            print(f"Frame {filename} saved.")
            current_frame += 1
            if current_frame >3 :
                break

cap.release()
cv2.destroyAllWindows()
