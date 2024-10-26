import torch
import torch.nn.functional as F
from configs import args


def test_cv(data_source, model):
    model.eval()
    total_loss = 0
    correct = 0

    data_iterator = data_source
    num_data = 0.0
    for batch_id, batch in enumerate(data_iterator):
        data, targets = batch
        data, targets = data.to(args.device), targets.to(args.device)

        output = model(data)
        total_loss += F.cross_entropy(output, targets, reduction='sum').item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        num_data += output.size(0)

    acc = float(correct) / float(num_data)
    total_l = total_loss / float(num_data)
    # print('___Test : epoch: {}: Average loss: {:.4f}, '
    #       'Accuracy: {}/{} ({:.4f}%)'.format(epoch, total_l, correct, num_data, acc))
    model.train()
    return total_l, acc


def test_poison_cv(helper, data_source, model, adversarial_index=-1):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    data_iterator = data_source
    num_data = 0.0
    for batch_id, batch in enumerate(data_iterator):

        if args.attack_mode.lower() in ['mr', 'dba']:
            data, target = helper.get_poison_batch(batch, evaluation=True)
        elif args.attack_mode.lower() == 'combine':
            data, target = helper.get_poison_batch(batch, evaluation=True, adversarial_index=adversarial_index)
        else:
            for pos in range(len(batch[0])):
                batch[1][pos] = helper.params['poison_label_swap']

            data, target = batch
        data = data.to(args.device)
        target = target.long().to(args.device)
        data.requires_grad_(False)
        target.requires_grad_(False)

        output = model(data)
        total_loss += F.cross_entropy(output, target, reduction='sum').data.item()  # sum up batch loss
        num_data += target.size(0)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

    acc = float(correct) / float(num_data)
    total_l = total_loss / float(num_data)
    # print(f"--- Test poisoned (Client {client_id}, Target label {helper.params['poison_label_swap']} ): "
    #       f"epoch: {epoch}: Average poisoned loss: {total_l:.4f}, Accuracy: {correct}/{num_data} ({acc:.0f}%)")
    model.train()
    return total_l, acc
