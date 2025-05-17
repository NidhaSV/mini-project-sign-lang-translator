import mediapipe as mp
import cv2
import numpy as np

COLOR_GREEN = (0,255,0)

'''
Detect and segment human hands in an image using OpenCV and a pre-trained MediaPipe model
    Args:
        image (cv2.typing.MatLike): The input image to be segmented

    Returns:
        None
'''
def segment_hand(image):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize the MediaPipe Hands model
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.75) as hands:
        # Process the image to detect hands
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            # No hand detected
            return
        
        # Create a mask for the segmented hand
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        all_landmarks = []
        # Loop through detected hands and create a convex hull around the landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in hand_landmarks.landmark]
            all_landmarks += landmarks
            hull = cv2.convexHull(np.array(landmarks))

            # Draw the convex hull on the mask
            cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
            
            # Draw hand landmarks on the original image (optional)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Apply the mask to segment the hand
        masked_img = cv2.bitwise_and(image, image, mask=mask)

        full_hull = cv2.convexHull(np.array(all_landmarks))
        # Get the bounding rectangle of the full convex hull
        x, y, w, h = cv2.boundingRect(full_hull)
        
        # Calculate the side length of the minimal enclosing square
        side_length = max(w, h)
        
        # Adjust the bounding box to be a square
        x_center, y_center = x + w // 2, y + h // 2
        x_square = max(x_center - side_length // 2, 0)
        y_square = max(y_center - side_length // 2, 0)
        x_square_end = min(x_square + side_length, image.shape[1])
        y_square_end = min(y_square + side_length, image.shape[0])
        
        cv2.rectangle(image,(x_square-1,y_square-1),(x_square_end+1, y_square_end+1),COLOR_GREEN,1)

        # Crop the image to the square region
        segmented_hand = masked_img[y_square:y_square_end, x_square:x_square_end]
        
        return segmented_hand