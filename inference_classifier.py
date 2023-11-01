import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import dlib
from pynput import keyboard

start_time = time.time()
condition_satisfied = False
detector = dlib.get_frontal_face_detector()



import tkinter as tk

# Global variable for minimum detected time
min_detected_time = 1.5 # start with 1 second

def update_detected_time(val):
    global min_detected_time
    min_detected_time = float(val)

# Create a tkinter window
root = tk.Tk()
root.title('Settings')

# Create the scale (slider)
scale = tk.Scale(root, from_= 0.75, to=3.0, resolution=0.25, length=200,
                 orient=tk.HORIZONTAL, command=update_detected_time)
scale.set(min_detected_time)
scale.pack()
from PIL import Image, ImageTk

def show_help():
    # Create a new window
    help_window = tk.Toplevel(root)
    help_window.title("Help")

    # Open the image file
    img = Image.open("dictionary.png")

    # Resize image if it's too big for the screen
    max_size = (500, 600)  # Max size (width, height)
    img.thumbnail(max_size)

    # Convert the Image object to a PhotoImage object (for Tkinter)
    img_tk = ImageTk.PhotoImage(img)

    # Create a label and add the image to it
    label = tk.Label(help_window, image=img_tk)

    # This line ensures the image is not garbage collected
    label.image = img_tk

    # Add the label to the window
    label.pack()

# Create a "Help" button
help_button = tk.Button(root, text="Help", command=show_help)
help_button.pack()





predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model_dict = pickle.load(open('model1000.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
cap.set(3, 1000)  # set frame width
cap.set(4, 800)  # set frame height

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

predicted_character = ''
character_counter = 0

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J' , 10: 'K', 11: 'L',
               12: 'M', 13:'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24:'Y', 25:'clear', 26:'delete', 27: 'space', 28:'Z' }

# Load chat icon image with alpha channel
chat_icon = cv2.imread("speech_bubble.png", -1)

# Real-time text input
text = ""
def on_press(key):
    global text
    try:
        text += key.char
    except AttributeError:
        if key == keyboard.Key.space:
            text += ' '
        elif key == keyboard.Key.backspace:
            text = text[:-1]
        elif key == keyboard.Key.esc:
            return False  # stop listener

# Start listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Load the face detector from dlib
detector = dlib.get_frontal_face_detector()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    # Update the Tkinter window
    root.update_idletasks()
    root.update()



    for face in faces:
        landmarks = predictor(image=gray, box=face)

        # Position of chat icon is near mouth, can be adjusted as needed
        chat_x = landmarks.part(24).x
        chat_y = landmarks.part(24).y

        # Resize icon to desired size and create mask
        width = 200
        height = 150  # width and height of the icon
        chat_icon_resized = cv2.resize(chat_icon, (width, height))
        mask = chat_icon_resized[:, :, 3]  # Extract the mask from the 4th channel
        mask_inv = cv2.bitwise_not(mask)  # Invert the mask
        chat_icon_resized = chat_icon_resized[:, :, 0:3]  # Remove the 4th channel

        # Checking bounds to make sure we don't exceed the frame
        start_y = max(0, chat_y)
        end_y = min(chat_y+height, frame.shape[0])
        start_x = max(0, chat_x)
        end_x = min(chat_x+width, frame.shape[1])

        # Only draw the icon if it fits within the frame
        if end_y > start_y and end_x > start_x:
            # Extract the region of interest (roi) from the frame
            roi = frame[start_y:end_y, start_x:end_x]

            # Resize mask and chat icon to roi size
            mask_resized = cv2.resize(mask, (end_x - start_x, end_y - start_y))
            mask_inv_resized = cv2.bitwise_not(mask_resized)
            chat_icon_roi = cv2.resize(chat_icon_resized, (end_x - start_x, end_y - start_y))

            # Black-out the area in roi where chat icon will be placed
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv_resized)

            # Take only the region of chat icon from chat_icon image
            chat_fg = cv2.bitwise_and(chat_icon_roi, chat_icon_roi, mask=mask_resized)

            # Place the chat icon in roi
            dst = cv2.add(roi_bg, chat_fg)
            frame[start_y:end_y, start_x:end_x] = dst

            # Break the text into lines if it exceeds the speech bubble's width
            max_width = 180  # Maximum width of a line in pixels
            max_text_len = 12  # Maximum length of a line in characters
            text_lines = [text[i:i+max_text_len] for i in range(0, len(text), max_text_len)]
            for i, line in enumerate(text_lines):
                y_text = start_y + 30 + i * 20  # adjust position and line height as needed
                cv2.putText(frame, line, (start_x + 40, y_text+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # only make a prediction if exactly one hand is detected
        if len(results.multi_hand_landmarks) == 1:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = labels_dict[int(prediction[0])]
            if predicted_character == predicted_label:
                # if it's the first time this character is predicted
                if start_time is None:
                    # store the start time
                    start_time = time.time()
            else:
                # if the character changed, reset the start time
                start_time = time.time()

            # if the character has been detected for more than one second
            if start_time is not None and time.time() - start_time > min_detected_time:
                start_time = None  # reset the start time for the next character

                if predicted_character == 'clear':
                    text = ''
                elif predicted_character == 'delete':
                    text = text[:-1]
                elif predicted_character == 'space':
                    text += ' '
                else:
                    text += predicted_character

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything is done, release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
