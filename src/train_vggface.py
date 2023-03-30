import os
from tqdm import tqdm
import argparse
import mediapipe as mp

import torch
import torch.nn as nn

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

    parser.add_argument('--model', type=str, default='mobilenet_v2')
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
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=args.face_threshold)
    load_datasets.prepare_crop_face(train_data_path, crop_train_data_path)  # train
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

    # loss function setting
    criterion = nn.CrossEntropyLoss()

    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model setting
    if args.model == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(pretrained=True)
    elif args.model == 'efficientnet_b0':
        model = 1
    elif args.model == 'swin-T':
        model = 1
    else:
        raise ValueError

    # model train
    set_parameter_requires_grad(model, requires_grad=False)
    #set_parameter_requires_grad(model.classifier, requires_grad=True)
    set_parameter_requires_grad(model.head.fc, requires_grad=True)
    train(model, device, train_loader, test_loader, criterion, train_row_all, test_row_all, n_epochs=1, learningrate=0.001, robust=False)
    set_parameter_requires_grad(model, requires_grad=True)
    train(model, device, train_loader, test_loader, criterion, train_row_all, test_row_all, n_epochs=10, learningrate=1e-4, robust=False)

    # validation
    epoch_val_accuracy,epoch_val_loss = test(model, device, test_loader, criterion, test_row_all)
    print(f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

    # save model
    torch.save(model, os.path.join(model_path, args.model + '_tr1.pt'))

    # del head(classifier) & save model
    model.head.fc=torch.nn.Identity()
    torch.save(model.state_dict(), os.path.join(model_path, args.model + '_state_tr1.pt'))
    torch.save(model, os.path.join(model_path, args.model + '_nohead_tr1.pt'))

    """
    from torchvision.models import resnet101,mobilenet_v2
    import timm
    #model=resnet101(pretrained=True)
    #model=mobilenet_v2(pretrained=True)
    #model=torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
    model=timm.create_model('rexnet_150', pretrained=True) #'vit_base_patch16_224' 'tf_efficientnet_b4_ns'
    #model=timm.create_model('tf_efficientnet_b2_ns', pretrained=True,features_only=True)
    print(model)
    
    #model.classifier=nn.Linear(in_features=1536, out_features=num_classes) #1792 #1536 #1280 #1408
    model.head.fc=nn.Linear(in_features=1920, out_features=num_classes)
    model=model.to(device)
    print(model)
    
    if True:
        img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        model=model.to(device)
        model.eval()
        f=model.forward(img)
        print(f.shape)
        model.train()
    """
