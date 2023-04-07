import time
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import os

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

# Draw Bounding box, headpose
def draw_bbox_axis(frame, face_pos):
    (x, y, x2, y2) = face_pos

    # BBox draw
    cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)),
                  color=(255, 255, 255), thickness=2)

# Draw Russell's Circumplex Model
def draw_russell(frame, valence, arousal, emotion):
    # TODO : x,y 명칭을 반대로했는데 언젠가 수정하자
    x_shape, y_shape, _ = frame.shape
    base_xy = 150
    len_xy = 120

    # Box 1
    add_image = np.zeros((299, y_shape, 3), np.uint8)
    add_image = cv2.rectangle(add_image, (base_xy - len_xy, base_xy - len_xy), (base_xy + len_xy, base_xy + len_xy),
                              color=(255, 255, 255), thickness=2)
    add_image = cv2.line(add_image, (base_xy - len_xy, base_xy), (base_xy + len_xy, base_xy), color=(255, 255, 255),
                         thickness=1)
    add_image = cv2.line(add_image, (base_xy, base_xy - len_xy), (base_xy, base_xy + len_xy), color=(255, 255, 255),
                         thickness=1)
    add_image = cv2.putText(add_image, 'Valence', (base_xy - 42, base_xy - len_xy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 1)
    add_image = cv2.putText(add_image, 'Arousal', (base_xy + len_xy + 5, base_xy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 1)

    valence_xy = int(base_xy + len_xy * valence)
    arousal_xy = int(base_xy - len_xy * arousal)  # Y축이라 마이너스 적용
    add_image = cv2.line(add_image,
                         (valence_xy, arousal_xy),
                         (valence_xy, arousal_xy),
                         color=(0, 0, 255), thickness=5)

    # Box 2
    box2_y_region = 600
    add_image = cv2.putText(add_image, 'Output', (y_shape - box2_y_region, base_xy - len_xy + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 2)
    add_image = cv2.putText(add_image, f'Valence : {str(valence)}', (y_shape - box2_y_region, base_xy - len_xy + 47),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    add_image = cv2.putText(add_image, f'Arousal : {str(arousal)}', (y_shape - box2_y_region, base_xy - len_xy + 69),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    add_image = cv2.putText(add_image, f'Emotion : {str(emotion)}', (y_shape - box2_y_region, base_xy - len_xy + 91),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    # Line
    add_image = cv2.line(add_image, (y_shape//2, 20), (y_shape//2, add_image.shape[1]-20), color=(255, 255, 255), thickness=2)

    frame = cv2.vconcat([frame, add_image])
    frame = cv2.resize(frame, (int(x_shape/frame.shape[0]*y_shape), x_shape))

    return frame

# Main
def main(model_path, img_size, mtl=False, save_video=False, save_path='demo.mp4'):

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

    # Save Video
    if save_video:
        width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        print((int(height / (height+299) * width), height)) # TODO : 여기 계산식도 draw_russell처럼 y,x 바뀐 상황
        out = cv2.VideoWriter(save_path, fourcc, 60, (int(height / (height+299) * width), height))

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
            if mtl:
                emotion = idx_to_class[np.argmax(scores[0:8])]
                valence = round(scores[8], 2)
                arousal = round(scores[9], 2)
                print(valence)
                print(arousal)
            else:
                emotion = idx_to_class[np.argmax(scores)]

            # Draw Image
            draw_bbox_axis(frame=frame, face_pos=(x, y, x2, y2))
            frame = cv2.putText(frame, f'Emotion : {emotion}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            if mtl:
                frame = cv2.putText(frame, f'Valence : {str(valence)}, Arousal : {str(arousal)}',
                                    (x, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                                    2)

            # Draw Russell's Circumplex Model
            if mtl:
                frame = draw_russell(frame, valence, arousal, emotion)

            # Show Image
            cv2.imshow("", frame)
            print(frame.shape)

            # Save Video
            if save_video:
                out.write(frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':

    img_size = 224

    # Model 2
    model_path = os.path.join(os.getcwd().split('/src')[0], 'models/enet_b0_8_va_mtl.pt')
    mtl = True

    # Save Video
    save_video = 1
    save_path = os.path.join(os.getcwd().split('/src')[0], 'demo.mp4')

    main(model_path = model_path,
         img_size = img_size,
         mtl = mtl,
         save_video = save_video,
         save_path = save_path)
