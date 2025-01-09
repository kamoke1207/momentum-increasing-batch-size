import torch
from .get_full_grad_list import get_full_grad_list


def train(epoch, steps, model, device, trainset, optimizer, criterion, batch_size, cuda):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        steps += 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 全勾配ノルムを計算
    p_norm = get_full_grad_list(model, trainset, optimizer, batch_size, cuda)
    norm_result = [epoch + 1, steps, p_norm]

    # 訓練結果を集計
    train_accuracy = 100. * correct / total
    train_result = [epoch + 1, steps, train_loss / (batch_idx + 1), train_accuracy]

    return steps, norm_result, train_result
