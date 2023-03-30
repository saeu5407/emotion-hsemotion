import os
import cv2
import mediapipe as mp
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# cropped one image
def crop_face_n_save(face_detection, image_path, save_path, target_size=(224, 224)):

    # load, preproc
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect face
    detected = face_detection.process(frame)
    if detected.detections:
        face_pos = detected.detections[0].location_data.relative_bounding_box

        # scale bbox
        x = int(frame.shape[1] * max(face_pos.xmin, 0))
        y = int(frame.shape[0] * max(face_pos.ymin, 0))
        w = int(frame.shape[1] * min(face_pos.width, 1))
        h = int(frame.shape[0] * min(face_pos.height, 1))

        face_plus_scalar = 20
        x2 = min(x + w + face_plus_scalar, frame.shape[1])
        y2 = min(y + h + face_plus_scalar, frame.shape[0])
        x = max(0, x - face_plus_scalar)
        y = max(0, y - face_plus_scalar)

        # output cropped face image
        frame = frame[y:y2, x:x2, :]
        frame = cv2.resize(frame, target_size)
        cv2.imwrite(save_path + os.path.basename(image_path), frame)
        return 1
    return 0

# crop face main
def prepare_crop_face(data_path, save_path):

    image_path_list = glob.glob(data_path + '/*' + 'png') + \
                      glob.glob(data_path + '/*' + 'jpg') + \
                      glob.glob(data_path + '/*' + 'jpeg')
    crop_cnt = 0
    for image_path in image_path_list:
        result = crop_face_n_save(face_detection, image_path, save_path, target_size=(224, 224))
        crop_cnt += result
    print(f"[{round(crop_cnt/len(image_path_list),2)}] total image length is {len(image_path_list)}, cropped {crop_cnt}")

# load dataset
def set_datasets(train_data_path, valid_data_path, target_size=(224, 224), batch_size=48, num_workers=4, use_cuda=True):

    train_transforms = transforms.Compose(
        [
            transforms.Resize(target_size),
            # transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(target_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_dataset = datasets.ImageFolder(root=valid_data_path, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    print(f"train_datasets's length : {len(train_dataset)}, test's length : {len(test_dataset)}")
    print(f"datasets's class length : {len(train_dataset.classes)}")

    return train_dataset, test_dataset, train_loader, test_loader

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser(description='Datasets Parameter')
    parser.add_argument('--data_name', type=str, default='vggface2')
    parser.add_argument('--data_path', type=str, default=os.getcwd().split(os.path.sep + 'src')[0] + 'datasets/')
    parser.add_argument('--face_threshold', type=float, default=0.9)
    parser.add_argument('--target_size', type=float, default=224)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_cuda', type=bool, default=True)
    args = parser.parse_args()

    # dataset path
    default_data_path = os.path.join(args.data_path, args.data_name)
    train_data_path = os.path.join(default_data_path, '')
    valid_data_path = os.path.join(default_data_path, 'val')
    crop_train_data_path = os.path.join(default_data_path, 'train_crop')
    crop_valid_data_path = os.path.join(default_data_path, 'val_crop')

    # prepare dataset
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=args.face_threshold)
    prepare_crop_face(train_data_path, crop_train_data_path) # train
    prepare_crop_face(valid_data_path, crop_valid_data_path) # valid

    # load dataset
    train_dataset, test_dataset, train_loader, test_loader = \
        set_datasets(train_data_path,
                     valid_data_path,
                     target_size=(args.target_size,args.target_size),
                     batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     use_cuda=args.use_cuda)