import cv2
import numpy as np
import mediapipe as mp
import subprocess
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a) # Start
    b = np.array(b) # Middle
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def process_video(input_path, output_path):
    # Rep information
    rep_data = []
    rep_start_time = None
    rep_count = 0
    last_rep_time = 0

    # Display
    display_text = "INIT"
    state_color = (255, 255, 255)
    display_rep_time = 0

    # Tracking
    state = "TOP"
    prev_hip_y = None
    start_hip_y = None
    descent_start_y = 0
    lowest_hip_y = 0
    descent_distance = 0
    hit_depth = False
    history = []
    display_rep_time = 0
    frame_index = 0
    eccentric_frames = 0

    # Left vs right
    current_side = None
    visibility_history = []

    # Constants
    MIN_REP_INTERVAL = 1.5
    MOVEMENT_THRESHOLD = 0.005
    TOP_THRESHOLD = 0.02
    MIN_DESCENT = 0.02
    MIN_ECCENTRIC_FRAMES = 3

    # Extract relevant landmarks
    relevant_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]

    # Color relevant landmarks and connections, blacken and shrink irrelevant landmarks and connections
    landmark_specs = {}

    for landmark in mp_pose.PoseLandmark:
        if landmark in relevant_landmarks:
            landmark_specs[landmark] = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 2, circle_radius = 2)
        else:
            landmark_specs[landmark] = mp_drawing.DrawingSpec(color = (0, 0, 0), thickness = 1, circle_radius = 1)

    connection_specs = {}

    for connection in mp_pose.POSE_CONNECTIONS:
        start_landmark = connection[0]
        end_landmark = connection[1]
        if start_landmark in relevant_landmarks and end_landmark in relevant_landmarks:
            connection_specs[connection] = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness = 2)
        else:
            connection_specs[connection] = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness = 1)
    # VIDEO FEED
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.3) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            # Track time
            frame_index += 1
            time = frame_index/fps

            # Change colors from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False #Saves Memory

            # Make detection
            results = pose.process(image)

            # Change back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates of hip and knee landmarks
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Hip and knee visibility
                r_hip_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
                r_knee_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
                l_hip_visibility = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
                l_knee_visibility = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility

                # Decide side used
                l_visibility = l_hip_visibility + l_knee_visibility
                r_visibility = r_hip_visibility + r_knee_visibility
                visibility_history.append((l_visibility, r_visibility))
                if len(visibility_history) > 15:
                    visibility_history.pop(0)
                
                l_avg = np.mean([v[0] for v in visibility_history])
                r_avg = np.mean([v[1] for v in visibility_history])

                if current_side == None:
                    current_side = "LEFT" if l_avg > r_avg else "RIGHT"
                elif current_side == "LEFT" and r_avg > l_avg + 0.15:
                    current_side = "RIGHT"
                elif current_side == "RIGHT" and l_avg > r_avg + 0.15:
                    current_side = "LEFT"

                hip = l_hip if current_side == "LEFT" else r_hip
                knee = l_knee if current_side == "LEFT" else r_knee
                hip_visibility = l_hip_visibility if current_side == "LEFT" else r_hip_visibility
                knee_visibility = l_knee_visibility if current_side == "LEFT" else r_knee_visibility

                thigh_length = np.linalg.norm(np.array(hip) - np.array(knee))
                top_of_knee = knee[1] - 0.08 * thigh_length # Slight offset to get top of knee instead of knee joint
                hip_crease = hip[1] - 0.09 * thigh_length # Slight offset to get hip crease
                knee[0] = (knee[0] - 0.08 * thigh_length) if current_side == "LEFT" else (knee[0] + 0.08 * thigh_length)
                hip[0] = (hip[0] - 0.09 * thigh_length) if current_side == "LEFT" else (hip[0] + 0.09 * thigh_length)

                # Get pixel values for visualization
                knee_top_pixel = (int(knee[0] * width), int(top_of_knee * height))
                hip_crease_pixel = (int(hip[0] * width), int(hip_crease * height))
                
                # Rep and depth tracking
                if hip_visibility > 0.5 and knee_visibility > 0.5:
                    
                    if prev_hip_y is None:
                        prev_hip_y = hip_crease
                        continue

                    alpha = 0.7 # Exponential moving average so smooth jitter
                    current_hip_y = alpha * hip_crease + (1 - alpha) * prev_hip_y

                    if start_hip_y == None:
                        start_hip_y = current_hip_y

                    # Movement triggers
                    history.append(current_hip_y)
                    if len(history) > 5: 
                        history.pop(0) # Moving window of last 5 frames
                    avg_hip_y = np.mean(history)
                    moving_down = current_hip_y > avg_hip_y + MOVEMENT_THRESHOLD # Compare current with recent trend
                    moving_up = current_hip_y < avg_hip_y - MOVEMENT_THRESHOLD

                    if state == "TOP":
                        display_text = "TOP"
                        state_color = (255, 255, 255)

                        if moving_down:
                            eccentric_frames +=1
                        else:
                            eccentric_frames = 0

                        if eccentric_frames >= MIN_ECCENTRIC_FRAMES:
                            state = "ECCENTRIC"
                            rep_start_time = time # Gets time 
                            hit_depth = False
                            descent_start_y = current_hip_y
                            lowest_hip_y = current_hip_y

                    elif state == "ECCENTRIC":
                        display_text = "ECCENTRIC"
                        state_color = (0, 255, 255)

                        lowest_hip_y = max(lowest_hip_y, current_hip_y)
                        descent_distance = lowest_hip_y - descent_start_y

                        if hip_crease > top_of_knee:
                            state = "BOTTOM"
                            hit_depth = True
                            eccentric_frames = 0

                        elif moving_up and descent_distance < MIN_DESCENT:
                            state = "TOP"
                            rep_start_time = None
                            eccentric_frames = 0
                        elif moving_up and descent_distance >= MIN_DESCENT:
                            state = "CONCENTRIC"

                    elif state == "BOTTOM":
                        display_text = "DEPTH"
                        state_color = (0, 255, 0)

                        if moving_up:
                            state = "CONCENTRIC"
                    
                    elif state == "CONCENTRIC":
                        display_text = "DEPTH" if hit_depth else "NO DEPTH"
                        state_color = (0, 255, 0) if hit_depth else (0, 0, 255)

                        if abs(current_hip_y - start_hip_y) < TOP_THRESHOLD:                            
                            if descent_distance > MIN_DESCENT:
                                if (time - last_rep_time) > MIN_REP_INTERVAL:
                                    rep_time = time - rep_start_time
                                    rep_data.append({
                                        "rep" : rep_count+1,
                                        "depth": "Hit depth" if hit_depth else "No depth",
                                        "time" : round(rep_time, 2)})
                                    rep_count += 1
                                    last_rep_time = time
                            state = "TOP"
                            rep_start_time = None
                            hit_depth = False

                    prev_hip_y = current_hip_y
                    
                    # Rep info display
                    cv2.putText(image, f'Reps: {rep_count}', (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if rep_start_time is not None:
                        display_rep_time = round(time - rep_start_time, 2)
                    cv2.putText(image, F'Time: -' if display_rep_time == 0 else F'Time: {display_rep_time}s', (50, 100),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, display_text, (50, 150),
                                cv2.FONT_HERSHEY_COMPLEX, 1, state_color, 2, cv2.LINE_AA)
                    
                    # Draw top of knee and hip crease for depth visualization
                    cv2.circle(image, knee_top_pixel, 4, (0, 255, 255), -1)
                    cv2.circle(image, hip_crease_pixel, 4, (255, 0, 255), -1)
                    depth_line_color = (0, 255, 0) if hit_depth else (0, 0, 255)
                    cv2.line(image, hip_crease_pixel, (knee_top_pixel[0], hip_crease_pixel[1]), depth_line_color, 2)
                    cv2.line(image, knee_top_pixel, (hip_crease_pixel[0], knee_top_pixel[1]), depth_line_color, 2)
                    
                    # Debug
                    print(
                        f"{state} | hip={current_hip_y:.3f} prev={prev_hip_y:.3f} | "
                        f"down={moving_down} up={moving_up} | "
                        f"depth={hit_depth} | "
                        f"desc={(lowest_hip_y - descent_start_y):.3f}")

            except Exception as e:
                #print("ERROR: ", e)
                pass

            # Render landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec = landmark_specs,
                                    connection_drawing_spec = connection_specs)

            out.write(image)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        final_mp4 = output_path.replace(".mp4", "_compressed.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", final_mp4])

        return rep_data, final_mp4
    
    