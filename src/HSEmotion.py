import time
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

idx_to_class = {0: 'Anger',
                1: 'Contempt',
                2: 'Disgust',
                3: 'Fear',
                4: 'Happiness',
                5: 'Neutral',
                6: 'Sadness',
                7: 'Surprise'}

# BBOX, HEADPOSE DRAW
def draw_bbox_axis(frame, face_pos):
    (x, y, x2, y2) = face_pos

    # BBox draw
    cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)),
                  color=(255, 255, 255), thickness=2)

# MAIN
def main(model_path, img_size):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    # Load Models
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model = model.to(device)
    model.eval()

    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.9)
    cap = cv2.VideoCapture(0)

    while 1:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 거울 모드
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        loop_start_time = time.time()
        detected = face_detection.process(rgb_img)
        print(">>> BlazeFace Use Time : {}".format(time.time() - loop_start_time))

        if detected.detections:
            face_pos = detected.detections[0].location_data.relative_bounding_box
            x = int(rgb_img.shape[1] * max(face_pos.xmin, 0))
            y = int(rgb_img.shape[0] * max(face_pos.ymin, 0))
            w = int(rgb_img.shape[1] * min(face_pos.width, 1))
            h = int(rgb_img.shape[0] * min(face_pos.height, 1))

            # face_pos 확정
            face_plus_scalar = 20
            x2 = min(x + w + face_plus_scalar, rgb_img.shape[1])
            y2 = min(y + h + face_plus_scalar, rgb_img.shape[0])
            x = max(0, x - face_plus_scalar)
            y = max(0, y - face_plus_scalar)

            # HSEmotion 적용
            face_img = frame[y:y2, x:x2, :]
            img_tensor = test_transforms(Image.fromarray(face_img))
            img_tensor.unsqueeze_(0)
            scores = model(img_tensor.to(device))
            scores = scores[0].data.cpu().numpy()
            emotion = idx_to_class[np.argmax(scores)]

            # Draw Image
            draw_bbox_axis(frame=frame, face_pos=(x, y, x2, y2))
            cv2.putText(frame, f'Emotion : {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                        2)
            cv2.imshow("", frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if __name__ == '__main__':

    model_path = '../models/enet_b0_8_best_afew.pt'
    img_size = 224

    main(model_path = model_path, img_size = img_size)