from FL_Backdoor_CV.models.edge_case_cnn import Net
from FL_Backdoor_CV.models.resnet import ResNet18
from FL_Backdoor_CV.models.resnet9 import ResNet9
from FL_Backdoor_CV.models.cnn import LeNet5
from configs import args


def create_model():
    num_classes = 0
    if args.dataset == 'cifar10':
        num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100
    if args.dataset == 'emnist':
        if args.emnist_style == 'digits':
            num_classes = 10
        if args.emnist_style == 'byclass':
            num_classes = 62
    if args.dataset == 'fmnist':
        num_classes = 10

    model = None
    if args.dataset == 'emnist':
        if args.emnist_style == 'digits':
            model = Net(num_classes=num_classes)
        if args.emnist_style == 'byclass':
            model = ResNet9(num_classes=num_classes)
    elif args.dataset == 'fmnist':
        model = LeNet5(num_classes=num_classes)
    else:
        model = ResNet18(num_classes=num_classes)
    return model.to(args.device)


if __name__ == '__main__':
    model = create_model()
    print(list(model.state_dict().keys()))

