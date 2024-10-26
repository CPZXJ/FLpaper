import copy
import random
import time

import torch

from FL_Backdoor_CV.roles.evaluation import test_poison_cv, test_cv
from configs import args

import torch.optim as optim


class Client:
    def __init__(self, client_id, local_data, local_data_size, malicious=False):
        self.client_id = client_id
        self.local_data = local_data
        self.local_data_size = local_data_size
        self.malicious = malicious
        self.server = None
        self.adversarial_index = None
        self.range_no_id = None

    def local_train(self, local_model, helper, epoch, criterion=torch.nn.CrossEntropyLoss()):
        local_data = self.local_data
        if self.malicious:
            # === clean part of malicious clients.
            # === we need to remove indices corresponding to the target label when attackers conduct local training. ===
            if self.range_no_id is None:
                range_no_id = list(range(len(helper.train_dataset)))

                if args.attack_mode.lower() in ['mr', 'dba', 'flip']:
                    for ind, x in enumerate(helper.train_dataset):
                        imge, label = x
                        if label == helper.params['poison_label_swap']:
                            range_no_id.remove(ind)

                    if args.dataset == 'cifar10':
                        if args.attack_mode.lower() == 'mr':
                            for image in helper.params['poison_images_test'] + \
                                         helper.params['poison_images']:
                                if image in range_no_id:
                                    range_no_id.remove(image)

                elif args.attack_mode.lower() == 'combine':
                    target_label = None
                    if self.adversarial_index == 0:
                        target_label = helper.params['poison_label_swaps'][0]
                    elif self.adversarial_index == 1:
                        target_label = helper.params['poison_label_swaps'][1]
                    elif self.adversarial_index == 2:
                        target_label = helper.params['poison_label_swaps'][2]
                    elif self.adversarial_index == 3:
                        target_label = helper.params['poison_label_swap']
                    for ind, x in enumerate(helper.train_dataset):
                        imge, label = x
                        if label == target_label:
                            range_no_id.remove(ind)

                    if args.dataset == 'cifar10':
                        if self.adversarial_index == 3:
                            for image in helper.params['poison_images_test'] + \
                                         helper.params['poison_images']:
                                if image in range_no_id:
                                    range_no_id.remove(image)

                elif args.attack_mode.lower() in ['edge_case', 'neurotoxin']:
                    pass

                random.shuffle(range_no_id)
                self.range_no_id = range_no_id

            # === malicious training ===
            poison_optimizer = optim.SGD(local_model.parameters(), lr=helper.params['poison_lr'],
                                         momentum=helper.params['poison_momentum'],
                                         weight_decay=helper.params['poison_decay'])
            if args.attack_mode.lower() in ['mr', 'dba', 'flip', 'combine']:
                for internal_epoch in range(1, 1 + helper.params['retrain_poison']):

                    indices = random.sample(self.range_no_id, args.batch_size - args.num_poisoned_samples)

                    if args.alternating_minimization:

                        for x in helper.poisoned_train_data:
                            inputs_p, labels_p = None, None
                            if args.attack_mode.lower() == 'mr':
                                inputs_p, labels_p = helper.get_poison_batch(x)
                            elif args.attack_mode.lower() in ['dba', 'combine']:
                                inputs_p, labels_p = helper.get_poison_batch(x, adversarial_index=self.adversarial_index)
                            elif args.attack_mode.lower() == 'flip':
                                inputs_p, labels_p = x
                                for pos in range(labels_p.size(0)):
                                    labels_p[pos] = helper.params['poison_label_swap']
                            poison_optimizer.zero_grad()
                            output = local_model(inputs_p.to(args.device))
                            loss = criterion(output, labels_p.to(args.device))
                            loss.backward()
                            poison_optimizer.step()
                            break

                        for x in helper.get_train(indices):
                            inputs_c, labels_c = x
                            poison_optimizer.zero_grad()
                            output = local_model(inputs_c.to(args.device))
                            loss = criterion(output, labels_c.to(args.device))
                            loss.backward()
                            poison_optimizer.step()
                            break

                    else:

                        for (x1, x2) in zip(helper.poisoned_train_data, helper.get_train(indices)):
                            inputs_p, labels_p = None, None
                            if args.attack_mode.lower() == 'mr':
                                inputs_p, labels_p = helper.get_poison_batch(x1)
                            elif args.attack_mode.lower() in ['dba', 'combine']:
                                inputs_p, labels_p = helper.get_poison_batch(x1, adversarial_index=self.adversarial_index)
                            elif args.attack_mode.lower() == 'flip':
                                inputs_p, labels_p = x1
                                for pos in range(labels_p.size(0)):
                                    labels_p[pos] = helper.params['poison_label_swap']

                            inputs_c, labels_c = x2
                            if args.attack_mode.lower() == 'flip':
                                for pos in range(labels_c.size(0)):
                                    if labels_c[pos] == 7:
                                        labels_c[pos] = helper.params['poison_label_swap']

                            inputs = torch.cat((inputs_p, inputs_c))
                            labels = torch.cat((labels_p, labels_c))
                            inputs, labels = inputs.to(args.device), labels.to(args.device)
                            poison_optimizer.zero_grad()
                            output = local_model(inputs)
                            loss = criterion(output, labels)
                            loss.backward()
                            poison_optimizer.step()

                            break

                    # === test poison ===
                    if args.show_process:
                        if args.attack_mode.lower() == 'combine':
                            poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model,
                                                                     self.adversarial_index)
                        else:
                            poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                        print(f"Malicious id: {self.client_id}, "
                              f"P o i s o n - N o w ! "
                              f"Epoch: {internal_epoch}, "
                              f"Local poison accuracy: {poison_acc: .4f}, "
                              f"Local poison loss: {poison_loss: .4f}.")
                    else:
                        if internal_epoch % helper.params['retrain_poison'] == 0:
                            if args.attack_mode.lower() == 'combine':
                                poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model, self.adversarial_index)
                            else:
                                poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                            print(f"Malicious id: {self.client_id}, "
                                  f"P o i s o n - N o w ! "
                                  f"Epoch: {internal_epoch}, "
                                  f"Local poison accuracy: {poison_acc: .4f}, "
                                  f"Local poison loss: {poison_loss: .4f}.")

            elif args.attack_mode.lower() in ['edge_case', 'neurotoxin']:
                # === get gradient mask use global model and clean data ===
                mask_grad_list = None
                if args.attack_mode.lower() == 'neurotoxin':
                    assert helper.params['gradmask_ratio'] != 1
                    num_clean_data = 150
                    subset_data_chunks = random.sample(helper.params['participant_clean_data'], num_clean_data)
                    sampled_data = [helper.benign_train_data[pos] for pos in subset_data_chunks]
                    mask_grad_list = helper.grad_mask_cv(helper, local_model, sampled_data, criterion,
                                                         ratio=helper.params['gradmask_ratio'])

                for internal_epoch in range(1, 1 + helper.params['retrain_poison']):
                    # === malicious train ===
                    indices = random.sample(self.range_no_id, args.batch_size - args.num_poisoned_samples)
                    for (x1, x2) in zip(helper.poisoned_train_data, helper.get_train(indices)):
                        inputs_p, labels_p = x1
                        inputs_c, labels_c = x2
                        inputs = torch.cat((inputs_p, inputs_c))
                        for pos in range(labels_p.size(0)):
                            labels_p[pos] = helper.params['poison_label_swap']
                        labels = torch.cat((labels_p, labels_c))
                        inputs, labels = inputs.to(args.device), labels.to(args.device)
                        poison_optimizer.zero_grad()
                        output = local_model(inputs)
                        loss = criterion(output, labels)
                        loss.backward()
                        if args.attack_mode.lower() == 'neurotoxin':
                            mask_grad_list_copy = iter(mask_grad_list)
                            for name, parms in local_model.named_parameters():
                                if parms.requires_grad:
                                    parms.grad = parms.grad * next(mask_grad_list_copy)
                        poison_optimizer.step()

                    if args.show_process:
                        poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                        print(f"Malicious id: {self.client_id}, "
                              f"P o i s o n - N o w ! "
                              f"Epoch: {internal_epoch}, "
                              f"Local poison accuracy: {poison_acc: .4f}, "
                              f"Local poison loss: {poison_loss: .4f}.")
                    else:
                        if internal_epoch % helper.params['retrain_poison'] == 0:
                            poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                            print(f"Malicious id: {self.client_id}, "
                                  f"P o i s o n - N o w ! "
                                  f"Epoch: {internal_epoch}, "
                                  f"Local poison accuracy: {poison_acc: .4f}, "
                                  f"Local poison loss: {poison_loss: .4f}.")

            # === malicious test ===
            test_loss, test_acc = test_cv(helper.benign_test_data, local_model)
            print(f"Malicious id: {self.client_id}, "
                  f"Test accuracy: {test_acc: .4f}, "
                  f"Test loss: {test_loss: .4f}.")
        else:
            # === optimizer and local epochs ===
            if not args.is_poison:
                if args.aggregation_rule.lower() in ['roseagg', 'avg', 'fltrust']:
                    if epoch > 500:
                        lr = args.local_lr * args.local_lr_decay ** ((epoch - 500) // args.decay_step)
                    else:
                        lr = args.local_lr
                elif args.aggregation_rule.lower() in ['flame', 'foolsgold', 'rlr']:
                    lr = args.local_lr
            else:
                lr = args.local_lr_min

            """
            lr = args.local_lr
            elif args.dataset.lower() == 'fmnist':
                if epoch > 200:
                    lr = lr * 0.993 ** (epoch - 200)
            """
            # else:
            #     lr_init = helper.params['lr']
            #     traget_lr = helper.params['target_lr']
            #
            #     if helper.params['dataset'] == 'emnist':
            #         lr = 0.0001
            #         if helper.params['emnist_style'] == 'byclass':
            #             if epoch <= 500:
            #                 lr = epoch * (traget_lr - lr_init) / 499.0 + lr_init - (traget_lr - lr_init) / 499.0
            #             else:
            #                 lr = epoch * (-traget_lr) / 1500 + traget_lr * 4.0 / 3.0
            #
            #                 if lr <= 0.0001:
            #                     lr = 0.0001
            #     else:
            #         if epoch <= 500:
            #             lr = epoch * (traget_lr - lr_init) / 499.0 + lr_init - (traget_lr - lr_init) / 499.0
            #         else:
            #             lr = epoch * (-traget_lr) / 1500 + traget_lr * 4.0 / 3.0
            #             if lr <= args.local_lr_min:
            #                 lr = args.local_lr_min
            optimizer = optim.SGD(local_model.parameters(), lr=lr,
                                  momentum=helper.params['momentum'],
                                  weight_decay=helper.params['decay'])
            epochs = helper.params['retrain_no_times']

            # === local training ===
            for _ in range(epochs):
                for inputs, labels in local_data:
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    optimizer.zero_grad()
                    loss = criterion(local_model(inputs), labels)
                    loss.backward()
                    optimizer.step()

            # === local test ===
            # test_loss, test_acc = test_cv(helper.benign_test_data, local_model)
            # print(f"Benign id: {self.client_id}, "
            #       f"N o r m a l - T r a i n i n g ! "
            #       f"Test accuracy: {test_acc: .4f}, "
            #       f"Test loss: {test_loss: .4f}.")

        return local_model
