import os
import cv2

# Change this to your preferred directory.
DATA_DIR = './data/28'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 1000

cap = cv2.VideoCapture(0)

print('Press "q" to start collecting data. Press "e" to end.')

counter = 0
collecting_data = False
while counter < dataset_size:
    ret, frame = cap.read()

    # Check if frame is captured correctly
    if ret:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)

        # Start collecting data if 'q' is pressed
        if key == ord('q'):
            collecting_data = True
            print("Started collecting data...")

        # Stop collecting data and exit the loop if 'e' is pressed
        if key == ord('e'):
            print("Ended collecting data.")
            break

        # Save the frame if we are in data collecting mode
        if collecting_data:
            cv2.imwrite(os.path.join(DATA_DIR, '{}.jpg'.format(counter)), frame)
            counter += 1

    else:
        print("Failed to capture frame. Skipping...")

cap.release()
cv2.destroyAllWindows()
