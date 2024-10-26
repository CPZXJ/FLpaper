import copy
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import EMNIST, FashionMNIST

from FL_Backdoor_CV.helper import Helper
from configs import args

random.seed(0)
np.random.seed(0)


class Customize_Dataset(Dataset):
    def __init__(self, X, Y, transform):
        self.train_data = X
        self.targets = Y
        self.transform = transform

    def __getitem__(self, index):
        data = self.train_data[index]
        target = self.targets[index]
        data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.train_data)


class ImageHelper(Helper):
    corpus = None

    def __init__(self, params):
        super(ImageHelper, self).__init__(params)

    # === loading distributed training set and a global testing set ===
    def load_data(self):
        # === data load ===
        if self.params['dataset'] == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_dataset = datasets.CIFAR10(self.params['data_folder'], train=True, download=True,
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR10(self.params['data_folder'], train=False, transform=transform_test)

        if self.params['dataset'] == 'emnist':
            if self.params['emnist_style'] == 'digits':
                self.train_dataset = EMNIST(self.params['data_folder'], split="digits", train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(28, padding=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
                self.test_dataset = EMNIST(self.params['data_folder'], split="digits", train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

            elif self.params['emnist_style'] == 'byclass':
                self.train_dataset = EMNIST(self.params['data_folder'], split="byclass", train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(28, padding=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
                self.test_dataset = EMNIST(self.params['data_folder'], split="byclass", train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

            elif self.params['emnist_style'] == 'letters':
                self.train_dataset = EMNIST(self.params['data_folder'], split="letters", train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(28, padding=2),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
                self.test_dataset = EMNIST(self.params['data_folder'], split="letters", train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

        if self.params['dataset'] == 'fmnist':
            self.train_dataset = datasets.FashionMNIST(self.params['data_folder'], train=True, download=False,
                                                       transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))
                                                       ]))
            self.test_dataset = datasets.FashionMNIST(self.params['data_folder'], train=False, download=False,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                      ]))

    def get_img_classes(self):
        img_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if self.params['dataset'] == 'cifar10':
                if self.params['is_poison'] and self.params['attack_mode'] in ['MR', 'COMBINE']:
                    if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                        continue
            if label in img_classes:
                img_classes[label].append(ind)
            else:
                img_classes[label] = [ind]
        return img_classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = self.get_img_classes()
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        print("Data split:")
        labels = np.array(self.train_dataset.targets)
        for i, client in per_participant_list.items():
            split = np.sum(labels[client].reshape(1, -1) == np.arange(no_classes).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

        return per_participant_list

    # === split the dataset in a non-iid (class imbalance) fashion ===
    def sample_class_imbalance_train_data(self, train=True, n_clients=100, classes_per_client=3, balance=0.99, verbose=True):
        image_classes = self.get_img_classes()
        no_classes = len(image_classes.keys())
        n_data = sum([len(value) for value in image_classes.values()])
        per_participant_list = defaultdict(list)
        if balance >= 1.0:
            data_per_client = [n_data // n_clients] * n_clients
            data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients
        else:
            fracs = balance ** np.linspace(0, n_clients - 1, n_clients)
            fracs /= np.sum(fracs)
            fracs = 0.1 / n_clients + (1 - 0.1) * fracs
            data_per_client = [np.floor(frac * n_data).astype('int') for frac in fracs]
            data_per_client[0] = n_data - sum(data_per_client[1:])
            data_per_client = data_per_client[::-1]
            data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]
        if sum(data_per_client) > n_data:
            print("Impossible Split")
            exit()
        for n in range(no_classes):
            random.shuffle(image_classes[n])
        for user in range(n_clients):
            budget = data_per_client[user]
            c = np.random.randint(no_classes)
            while budget > 0:
                take = min(data_per_client_per_class[user], len(image_classes[c]), budget)
                sampled_list = image_classes[c][:take]
                per_participant_list[user].extend(sampled_list)
                image_classes[c] = image_classes[c][take:]
                budget -= take
                c = (c + 1) % no_classes

        def print_split():
            print("Data split:")
            labels = np.array(self.train_dataset.targets)
            for i, client in per_participant_list.items():
                split = np.sum(labels[client].reshape(1, -1) == np.arange(no_classes).reshape(-1, 1), axis=1)
                print(" - Client {}: {}".format(i, split))
            print()

        if verbose:
            print_split()
        return per_participant_list

    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False)
        return test_loader

    def load_distributed_data(self):
        # === sample indices for participants using Dirichlet distribution ===
        if self.params['class_imbalance']:
            indices_per_participant = self.sample_class_imbalance_train_data(
                n_clients=self.params['participant_population'],
                classes_per_client=self.params['classes_per_client'],
                balance=self.params['balance'])
        else:
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['participant_population'],
                alpha=self.params['dirichlet_alpha'])

        # === divide the training set into {self.params['participant_population']} clients ===
        train_loaders = [self.get_train(indices) for _, indices in indices_per_participant.items()]
        self.local_data_sizes = [len(indices) for _, indices in indices_per_participant.items()]
        self.train_data = train_loaders
        self.test_data = self.get_test()

    def load_root_dataset(self, samples=100):
        img_classes = self.get_img_classes()
        img_per_class = samples // len(img_classes)
        indices = list()
        for i, idxs in img_classes.items():
            indices.extend(random.sample(idxs, img_per_class))
        return self.get_train(indices)

    def load_benign_data(self):
        if self.params['dataset'] == 'cifar10' or \
                self.params['dataset'] == 'emnist' or \
                self.params['dataset'] == 'fmnist':
            if self.params['is_poison']:
                self.params['adversary_list'] = list(range(self.params['number_of_adversaries']))
            else:
                self.params['adversary_list'] = list()
            self.benign_train_data = self.train_data
            self.benign_test_data = self.test_data
        else:
            raise ValueError('Unrecognized dataset')

    def load_poison_data(self):
        if self.params['dataset'] == 'cifar10' or \
                self.params['dataset'] == 'emnist' or \
                self.params['dataset'] == 'fmnist':
            self.poisoned_train_data = self.poison_dataset()
            self.poisoned_test_data = self.poison_test_dataset()

        else:
            raise ValueError('Unrecognized dataset')

    def sample_poison_data(self, target_class):
        cifar_poison_classes_ind = []
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label == target_class:
                cifar_poison_classes_ind.append(ind)
        return cifar_poison_classes_ind

    def poison_dataset(self):
        indices = list()
        if args.attack_mode.lower() in ['mr', 'dba', 'flip', 'combine']:
            if args.attack_mode == 'MR':
                print(f"A T T A C K - M O D E: M o d e l - R e p l a c e m e n t !")
            elif args.attack_mode == 'DBA':
                print(f"A T T A C K - M O D E: D i s t r i b u t e d - B a c k d o o r - A t t a c k !")
            elif args.attack_mode == 'FLIP':
                print(f"A T T A C K - M O D E: F L I P !")
            elif args.attack_mode == 'COMBINE':
                print(f"A T T A C K - M O D E: C O M B I N E !")
            range_no_id = None
            if args.attack_mode.lower() in ['mr', 'dba']:
                range_no_id = list(range(len(self.test_dataset)))
                remove_no_id = self.sample_poison_data(self.params['poison_label_swap'])
                range_no_id = list(set(range_no_id) - set(remove_no_id))
            elif args.attack_mode.lower() == 'combine':
                range_no_id = list(range(len(self.test_dataset)))
                for i in self.params['poison_label_swaps'] + [self.params['poison_label_swap']]:
                    remove_no_id = self.sample_poison_data(i)
                    range_no_id = list(set(range_no_id) - set(remove_no_id))
            elif args.attack_mode.lower() == 'flip':
                range_no_id = self.sample_poison_data(7)
            while len(indices) < self.params['size_of_secret_dataset']:
                range_iter = random.sample(range_no_id, self.params['batch_size'])
                indices.extend(range_iter)
            self.poison_images_ind = indices
            return torch.utils.data.DataLoader(self.test_dataset,
                                               batch_size=args.num_poisoned_samples,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   self.poison_images_ind)
                                               )
        elif args.attack_mode.lower() in ['edge_case', 'neurotoxin']:
            if args.attack_mode.lower() == 'edge_case':
                print(f"A T T A C K - M O D E: E d g e - C a s e - B a c k d o o r !")
            elif args.attack_mode.lower() == 'edge_case':
                print(f"A T T A C K - M O D E: N E U R O T O X I N !")
            if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100':
                # === Load attackers training and testing data, which are different data ===
                with open('../data/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('../data/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)

                print('shape of edge case train data (southwest airplane dataset train)',
                      saved_southwest_dataset_train.shape)
                print('shape of edge case test data (southwest airplane dataset test)',
                      saved_southwest_dataset_test.shape)

                sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype=int)
                sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype=int)
                print(np.max(saved_southwest_dataset_train))

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                trainset = Customize_Dataset(X=saved_southwest_dataset_train, Y=sampled_targets_array_train,
                                             transform=transform)
                self.poisoned_train_loader = DataLoader(dataset=trainset, batch_size=args.num_poisoned_samples,
                                                        shuffle=True)

                testset = Customize_Dataset(X=saved_southwest_dataset_test, Y=sampled_targets_array_test,
                                            transform=transform)
                self.poisoned_test_loader = DataLoader(dataset=testset, batch_size=self.params['batch_size'],
                                                       shuffle=True)

                return self.poisoned_train_loader

            if self.params['dataset'] in ['emnist', 'fmnist']:
                # Load attackers training and testing data, which are different
                ardis_images = np.loadtxt('../data/ARDIS/ARDIS_train_2828.csv', dtype='float')
                ardis_labels = np.loadtxt('../data/ARDIS/ARDIS_train_labels.csv', dtype='float')

                ardis_test_images = np.loadtxt('../data/ARDIS/ARDIS_test_2828.csv', dtype='float')
                ardis_test_labels = np.loadtxt('../data/ARDIS/ARDIS_test_labels.csv', dtype='float')
                print(ardis_images.shape, ardis_labels.shape)

                # reshape to be [samples][width][height]
                ardis_images = ardis_images.reshape((ardis_images.shape[0], 28, 28)).astype('float32')
                ardis_test_images = ardis_test_images.reshape((ardis_test_images.shape[0], 28, 28)).astype('float32')

                # labels are one-hot encoded
                indices_seven = np.where(ardis_labels[:, 7] == 1)[0]
                images_seven = ardis_images[indices_seven, :]
                images_seven = torch.tensor(images_seven).type(torch.uint8)

                indices_test_seven = np.where(ardis_test_labels[:, 7] == 1)[0]
                images_test_seven = ardis_test_images[indices_test_seven, :]
                images_test_seven = torch.tensor(images_test_seven).type(torch.uint8)

                labels_seven = torch.tensor([7 for y in ardis_labels])
                labels_test_seven = torch.tensor([7 for y in ardis_test_labels])

                if args.dataset == 'emnist':

                    ardis_dataset = EMNIST(self.params['data_folder'], split="digits", train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

                    ardis_test_dataset = EMNIST(self.params['data_folder'], split="digits", train=False, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))
                                                ]))
                elif args.dataset == 'fmnist':
                    ardis_dataset = FashionMNIST(self.params['data_folder'], train=True, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))
                    ardis_test_dataset = FashionMNIST(self.params['data_folder'], train=False, download=True,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))

                ardis_dataset.data = images_seven
                ardis_dataset.targets = labels_seven

                ardis_test_dataset.data = images_test_seven
                ardis_test_dataset.targets = labels_test_seven

                print(images_seven.size(), labels_seven.size())

                self.poisoned_train_loader = DataLoader(dataset=ardis_dataset, batch_size=args.num_poisoned_samples,
                                                        shuffle=True)
                self.poisoned_test_loader = DataLoader(dataset=ardis_test_dataset,
                                                       batch_size=self.params['test_batch_size'], shuffle=True)

                return self.poisoned_train_loader
        else:
            raise ValueError("U n k n o w n - a t t a c k - m o d e l !")

    def get_poison_batch(self, batch, evaluation=False, adversarial_index=-1):
        inputs, labels = batch
        new_inputs = inputs
        new_labels = labels
        for index in range(len(new_inputs)):
            new_labels[index] = self.params['poison_label_swap']
            if evaluation:
                if args.attack_mode.lower() == 'mr':
                    if args.dataset == 'cifar10':
                        new_inputs[index] = self.train_dataset[
                            random.choice(self.params['poison_images_test'])][0]
                        new_inputs[index].add_(torch.FloatTensor(new_inputs[index].shape).normal_(0, 0.01))
                    else:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], -1)
                elif args.attack_mode.lower() == 'dba':
                    new_inputs[index] = self.add_pixel_pattern(inputs[index], adversarial_index)
                elif args.attack_mode.lower() == 'combine':
                    if adversarial_index == 0:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], 0)
                        new_labels[index] = self.params['poison_label_swaps'][0]
                    elif adversarial_index == 1:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], 1)
                        new_labels[index] = self.params['poison_label_swaps'][1]
                    elif adversarial_index == 2:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], 2)
                        new_labels[index] = self.params['poison_label_swaps'][2]
                    elif adversarial_index == 3:
                        if args.dataset == 'cifar10':
                            new_inputs[index] = self.train_dataset[
                                random.choice(self.params['poison_images_test'])][0]
                            new_inputs[index].add_(torch.FloatTensor(new_inputs[index].shape).normal_(0, 0.01))
                        else:
                            new_inputs[index] = self.add_pixel_pattern(inputs[index], 3)

                    else:
                        raise ValueError('Unrecognized Adversarial Index')
            else:
                if args.attack_mode.lower() == 'mr':
                    if args.dataset == 'cifar10':
                        new_inputs[index] = self.train_dataset[
                            random.choice(self.params['poison_images'])][0]
                        new_inputs[index].add_(torch.FloatTensor(new_inputs[index].shape).normal_(0, 0.01))
                    else:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], -1)
                elif args.attack_mode.lower() == 'dba':
                    new_inputs[index] = self.add_pixel_pattern(inputs[index], adversarial_index)
                elif args.attack_mode.lower() == 'combine':
                    if adversarial_index == 0:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], 0)
                        new_labels[index] = self.params['poison_label_swaps'][0]
                    elif adversarial_index == 1:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], 1)
                        new_labels[index] = self.params['poison_label_swaps'][1]
                    elif adversarial_index == 2:
                        new_inputs[index] = self.add_pixel_pattern(inputs[index], 2)
                        new_labels[index] = self.params['poison_label_swaps'][2]
                    elif adversarial_index == 3:
                        if args.dataset == 'cifar10':
                            new_inputs[index] = self.train_dataset[
                                random.choice(self.params['poison_images'])][0]
                            new_inputs[index].add_(torch.FloatTensor(new_inputs[index].shape).normal_(0, 0.01))
                        else:
                            new_inputs[index] = self.add_pixel_pattern(inputs[index], 3)
                    else:
                        raise ValueError('Unrecognized Adversarial Index')
        new_inputs = new_inputs
        new_labels = new_labels
        if evaluation:
            new_inputs.requires_grad_(False)
            new_labels.requires_grad_(False)
        return new_inputs, new_labels

    def add_pixel_pattern(self, ori_image, adversarial_index=-1):
        image = copy.deepcopy(ori_image)
        poison_patterns = []
        if adversarial_index == -1:
            for i in range(0, args.dba_trigger_num):
                poison_patterns = poison_patterns + self.params[str(i) + '_poison_pattern']
        else:
            poison_patterns = self.params[str(adversarial_index) + '_poison_pattern']
        if args.dataset == 'cifar10':
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 0.4914 * 10
                image[1][pos[0]][pos[1]] = 0.4822 / 5
                image[2][pos[0]][pos[1]] = 0.4465 / 5
        elif args.dataset in ['emnist', 'fmnist']:
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 0.1307 * 30
        return image

    def poison_test_dataset(self):
        if args.attack_mode.lower() in ['edge_case', 'neurotoxin']:
            return self.poisoned_test_loader
        else:
            return torch.utils.data.DataLoader(self.test_dataset,
                                               batch_size=self.params['test_batch_size'],
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   self.poison_images_ind
                                               ))

    def get_poison_test(self, indices):
        train_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices))
        return train_loader



