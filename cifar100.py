'''Train CIFAR100 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import json
from training import train, test
from utils import checkpoint, select_model, get_bs_scheduler, get_config_value, save_to_csv
from utils.setup_optimizer import setup_optimizer

# Command Line Argument
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training with Schedulers')
    parser.add_argument('config_path', type=str, help='path of config file(.json)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    lr = get_config_value(config, "init_lr")
    optimizer_name = get_config_value(config, "optimizer")
    epochs = get_config_value(config, "epochs")
    checkpoint_path = config.get("checkpoint_path", "checkpoint.pth.tar")
    csv_path = get_config_value(config, "csv_path")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Dataset Preparation
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Device Setting
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_name = get_config_value(config, "model")
    model = select_model(model_name=model_name, num_classes=100).to(device)
    print(f"model: {model_name}")

    criterion = nn.CrossEntropyLoss()

    optimizer = setup_optimizer(optimizer_name, model.parameters(), lr=lr)
    print(f"Optimizer: {optimizer_name}, Learning Rate: {lr}")

    bs_scheduler = get_bs_scheduler(config, trainset_length=len(trainset))

    # Lists to save results
    train_results = []
    test_results = []
    norm_results = []

    if args.resume:
        checkpoint_ = checkpoint.load(checkpoint_path)
        model.load_state_dict(checkpoint_.get('model_state_dict', model.state_dict()))
        optimizer.load_state_dict(checkpoint_.get('optimizer_state_dict', optimizer.state_dict()))
        bs_scheduler.load_state_dict(checkpoint_.get('bs_scheduler_state_dict', bs_scheduler.state_dict()))
        start_epoch = checkpoint_['epoch'] + 1
        train_results = checkpoint_.get('train_results', [])
        test_results = checkpoint_.get('test_results', [])
        norm_results = checkpoint_.get('norm_results', [])
        steps = checkpoint_['steps']
    else:
        start_epoch = 0
        steps = 0
        batch_size = bs_scheduler.get_batch_size()

    for epoch in range(start_epoch, epochs):
        batch_size = bs_scheduler.get_batch_size()
        print(f'batch size: {batch_size}')
        steps, norm_result, train_result = train(epoch, steps, model, device, trainset, optimizer, criterion, batch_size, cuda=args.cuda_device)
        norm_results.append(norm_result)
        train_results.append(train_result)

        test_result = test(epoch, model, device, testloader, criterion)
        test_results.append(test_result)

        bs_scheduler.step()

        checkpoint.save({
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bs_scheduler_state_dict': bs_scheduler.state_dict(),
            'train_results': train_results,
            'test_results': test_results,
            'norm_results': norm_results,
        }, checkpoint_path)

        print(f'Epoch: {epoch + 1}, Steps: {steps}, Train Loss: {train_results[epoch][2]:.4f}, Test Acc: {test_results[epoch][2]:.2f}%')

    # Save to CSV file
    save_to_csv(csv_path, {
        "train": train_results,
        "test": test_results,
        "norm": norm_results,
    })
