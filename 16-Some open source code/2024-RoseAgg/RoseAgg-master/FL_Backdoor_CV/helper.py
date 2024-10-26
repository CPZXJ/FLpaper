import math
import random

import numpy as np
import torch

from configs import args

# ========= Fix the random seed (Start) =========
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(0)
np.random.seed(0)


# ========= Fix the random seed (End) =========


class Helper:
    def __init__(self, params):
        self.target_model = None
        self.local_model = None
        self.training_data = None
        self.benign_test_data = None
        self.poisoned_data = None
        self.poisoned_test_data = None
        self.params = params
        self.best_loss = math.inf

    @staticmethod
    def get_weight_difference(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))
        difference_flat = torch.cat(res)
        return difference, difference_flat

    @staticmethod
    def get_l2_norm(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))
        difference_flat = torch.cat(res)
        l2_norm = torch.norm(difference_flat.clone().detach().to(args.device))
        l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())
        return l2_norm, l2_norm_np

    @staticmethod
    def clip_grad(norm_bound, weight_difference, difference_flat):
        l2_norm = torch.norm(difference_flat.clone().detach().to(args.device))
        scale = max(1.0, float(torch.abs(l2_norm / norm_bound)))
        for name in weight_difference.keys():
            weight_difference[name].div_(scale)
        return weight_difference, l2_norm

    @staticmethod
    def grad_mask_cv(helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()

        # ========= local training (Start) =========
        for participant_id in range(len(dataset_clearn)):
            train_data = dataset_clearn[participant_id]
            for inputs, labels in train_data:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward(retain_graph=True)
        # ========= local training (End) =========

        mask_grad_list = []
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))

                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                    k_layer += 1

            grad_list = torch.cat(grad_list).to(args.device)
            _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).to(args.device)
            mask_flat_all_layer[indices] = 1.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length].to(args.device)
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(args.device))

                    count += gradients_length

                    percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0

                    percentage_mask_list.append(percentage_mask1)

                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer] / np.sum(grad_abs_sum_list))

                    k_layer += 1
        else:
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().to(args.device)) / float(
                        len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1 * gradients, int(gradients_length * 1.0))
                    else:

                        _, indices = torch.topk(-1 * gradients, int(gradients_length * ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(args.device))

                    percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0

                    percentage_mask_list.append(percentage_mask1)

                    k_layer += 1

        model.zero_grad()
        return mask_grad_list

    def lr_decay(self, epoch):
        return 1

    def average_shrink_models(self, weight_accumulator, target_model, epoch, wandb):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = self.lr_decay(epoch)
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                print('skipping')
                continue
            update_per_layer = weight_accumulator[name] * (1 / self.params['partipant_sample_size']) * lr
            update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)
            data.add_(update_per_layer.to(args.device))
        return True

