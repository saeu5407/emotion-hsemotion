import os
from tqdm import tqdm
import argparse
import mediapipe as mp

import torch
import torch.nn as nn

import timm

from train.optimizer import RobustOptimizer
import train.load_datasets as load_datasets
from train.func_to_train import train, test

# freeze 함수
# adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

print(f"Torch: {torch.__version__}")

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser(description='Train Parameter')
    parser.add_argument('--data_name', type=str, default='vggface2')
    parser.add_argument('--data_path', type=str, default=os.getcwd().split(os.path.sep + 'src')[0])
    parser.add_argument('--face_threshold', type=float, default=0.9)
    parser.add_argument('--target_size', type=float, default=224)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--model', type=str, default='swin_T')
    parser.add_argument('--prepare', type=bool, default=1, help='if already prepare cropped dataset, use 1 or True')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # dataset path
    default_data_path = os.path.join(args.data_path, 'datasets', args.data_name)
    train_data_path = os.path.join(default_data_path, '')
    valid_data_path = os.path.join(default_data_path, 'val')
    crop_train_data_path = os.path.join(default_data_path, 'train_crop')
    crop_valid_data_path = os.path.join(default_data_path, 'val_crop')
    model_path = os.path.join(args.data_path, 'models')

    # prepare dataset
    if args.prepare == 0:
        face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=args.face_threshold)
        load_datasets.prepare_crop_face(train_data_path, crop_train_data_path)  # default
        load_datasets.prepare_crop_face(valid_data_path, crop_valid_data_path)  # valid

    # load dataset
    train_dataset, test_dataset, train_loader, test_loader = \
        load_datasets.set_datasets(train_data_path,
                                   valid_data_path,
                                   target_size=(args.target_size, args.target_size),
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   use_cuda=use_cuda)
    train_row_all = len(train_dataset)
    test_row_all = len(test_dataset)
    num_classes = len(train_dataset.classes)

    # loss function setting
    criterion = nn.CrossEntropyLoss()

    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model setting
    # if use other model search timm.list_models()
    if args.model == 'mobilenet_v2':
        model = timm.create_model('mobilenetv2_100', num_classes=num_classes, pretrained=True)
    elif args.model == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', num_classes=num_classes, pretrained=True)
    elif args.model == 'swin_T':
        model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=num_classes, pretrained=True)
    else:
        raise ValueError

    # model default
    set_parameter_requires_grad(model, requires_grad=False)
    if args.model == 'swin_T':
        set_parameter_requires_grad(model.head.fc, requires_grad=True)
    else:
        set_parameter_requires_grad(model.classifier, requires_grad=True)
    train(model, device, train_loader, test_loader, criterion, train_row_all, test_row_all, n_epochs=1, learningrate=0.001, robust=False)
    set_parameter_requires_grad(model, requires_grad=True)
    train(model, device, train_loader, test_loader, criterion, train_row_all, test_row_all, n_epochs=10, learningrate=1e-4, robust=False)

    # validation
    epoch_val_accuracy,epoch_val_loss = test(model, device, test_loader, criterion, test_row_all)
    print(f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

    # save model
    torch.save(model, os.path.join(model_path, args.model + '_pretrained_vgg.pt'))

    # del head(classifier) & save model
    if args.model == 'swin_T':
        model.head = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()
    torch.save(model.state_dict(), os.path.join(model_path, args.model + '_pretrained_vgg_state.pt'))
    torch.save(model, os.path.join(model_path, args.model + '_pretrained_vgg_nohead.pt'))

    print("Done.")