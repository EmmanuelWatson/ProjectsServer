import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

haarCascade = cv2.CascadeClassifier('harrcascade.xml')

# Initialize drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize webcam capture
cap = cv2.VideoCapture(0)
pTime = 0
cTime = time.time()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


def calculate_head_pose(landmarks, image_shape):
    # Define indices for landmarks around the face
    eye_left_idx = 33  # Left eye inner corner
    eye_right_idx = 263  # Right eye inner corner
    nose_tip_idx = 1  # Nose tip

    # Extract landmark positions
    eye_left = landmarks[eye_left_idx]
    eye_right = landmarks[eye_right_idx]
    nose_tip = landmarks[nose_tip_idx]

    # Calculate center of eyes and center of the face
    eyes_center_x = (eye_left.x + eye_right.x) / 2
    eyes_center_y = (eye_left.y + eye_right.y) / 2
    face_center_x = nose_tip.x
    face_center_y = nose_tip.y

    # Convert to pixel coordinates
    image_width, image_height = image_shape[1], image_shape[0]
    eyes_center = (int(eyes_center_x * image_width), int(eyes_center_y * image_height))
    face_center = (int(face_center_x * image_width), int(face_center_y * image_height))

    # Calculate differences
    dx = face_center[0] - eyes_center[0]
    dy = face_center[1] - eyes_center[1]

    # Determine direction
    direction = "Looking Forward"
    if abs(dx) > 30:
        if dx > 0:
            direction = "Turned Right"
        else:
            direction = "Turned Left"
    elif abs(dy) > 30:
        if dy > 0:
            direction = "Tilted Down"
        else:
            direction = "Tilted Up"

    return direction, face_center

with mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        faceRect = haarCascade.detectMultiScale(image ,scaleFactor=1.1 ,minNeighbors=4  )
        
        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Calculate and display head pose
                position, face_center = calculate_head_pose(face_landmarks.landmark, image.shape)
                
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                
                # Display movement information
                text_position = (10, 30 + (idx * 60))
                cv2.putText(image, f'Face {idx + 1}: {position}', text_position,
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Draw a circle around the detected face center
                #cv2.circle(image, face_center, 5, (255, 0, 0), -1)
                for (x,y,w,h) in faceRect :
                    cv2.rectangle(image ,(x,y) , (x+w , y+h) , (0,0,255) , thickness=1)

                
                # fps = 1/(cTime - pTime)
                # pTime = cTime

                # #displaying fps 
                # cv2.putText(image , f'FPS :{int(fps)}' , (0,480) ,cv2.FONT_HERSHEY_PLAIN , 2, (0,255,0) , 3)
        
        # Display the result
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
