import argparse

parser = argparse.ArgumentParser(description='PPDL')

# === dataset, data partitioning mode, device, model, and rounds
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'emnist', 'fmnist'],
                    help='dataset')

parser.add_argument('--params', default='utils/cifar10_params.yaml',
                    dest='params')

parser.add_argument('--emnist_style', default='digits', type=str,
                    help='byclass digits letters')

parser.add_argument('--class_imbalance', type=int, default=0,
                    help='split the dataset in a non-iid (class imbalance) fashion')

parser.add_argument('--balance', type=float, default=0.99,
                    help='balance of the data size')

parser.add_argument('--classes_per_client', type=int, default=2,
                    help='class per client')

parser.add_argument('--resume', default=0, type=int,
                    help='resume or not')

parser.add_argument('--resumed_name', default=None, type=str)

parser.add_argument('--rounds', default=2000, type=int,
                    help='total rounds for convergence')

# === client selection mode ===
parser.add_argument('--participant_population', default=500, type=int,
                    help='total clients')

parser.add_argument('--participant_sample_size', default=100, type=int,
                    help='participants each round')

parser.add_argument('--is_poison', default=0, type=int,
                    help='poison or not')

parser.add_argument('--number_of_adversaries', default=1, type=int,
                    help='the number of attackers')

parser.add_argument('--random_compromise', default=0, type=int,
                    help='randomly compromise benign client')

parser.add_argument('--poison_rounds', default=None, type=str,
                    help='e.g., 3001,3002,3003, conduct model poisoning at the 3001st, 3002nd, and 3003rd rounds')

parser.add_argument('--retrain_rounds', default=60, type=int,
                    help='continue to train {retrain_rounds} rounds starting when resume is true')

parser.add_argument('--poison_prob', type=float, default=0,
                    help='poison probability each round')

# === aggregation rule on the server ===
parser.add_argument('--aggregation_rule', default='avg', type=str,
                    choices=['avg', 'rlr', 'flame', 'foolsgold', 'roseagg', 'fltrust', 'fedcie'],
                    help='aggregation method')

parser.add_argument('--device', default='cuda:1', type=str,
                    help='device')

# === configuration of local training ===
parser.add_argument('--local_lr', type=float, default=0.1,
                    help='learning rate')

parser.add_argument('--local_lr_decay', type=float, default=0.991,
                    help='learning rate decay')

parser.add_argument('--decay_step', type=int, default=5)

parser.add_argument('--local_lr_min', type=float, default=0.001,
                    help='')

parser.add_argument('--global_lr', type=float, default=1,
                    help='')

parser.add_argument('--global_lr_decay', type=float, default=1,
                    help='')

parser.add_argument('--batch_size', type=int, default=64,
                    help='local batch size')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')

parser.add_argument('--decay', type=float, default=5e-4,
                    help='SGD weight_decay')

# === attack mode ===
parser.add_argument('--attack_mode', default='MR', type=str,
                    help='aggregation method, [MR, DBA, EDGE_CASE, FLIP, NEUROTOXIN, COMBINE]')

parser.add_argument('--num_poisoned_samples', default=6, type=int,
                    help='the number of poisoned samples in one batch')

parser.add_argument('--dba_trigger_num', default=4, type=int,
                    help='the number of distributed triggers')

parser.add_argument('--dba_poison_rounds',
                    default='2001,2003,2005,2007,2011,2013,2015,2017,2021,2023,2025,2027,'
                            '2031,2033,2035,2037,2041,2043,2045,2047,2051,2053,2055,2057',
                    type=str,
                    help="if {attack_mode} == 'DBA', the poison rounds is {dba_poison_rounds}")

parser.add_argument('--mal_boost', default=0, type=int,
                    help='scale up the poisoned model update')

parser.add_argument('--gradmask_ratio', default=0.5, type=float,
                    help='The proportion of the gradient retained in GradMask')

parser.add_argument('--multi_objective_num', default=4, type=int,
                    help='The number of injected backdoors. Default: wall ---> 2, pixel ---> 3')

parser.add_argument('--alternating_minimization', default=0, type=int,
                    help='')

# === save model ===
parser.add_argument('--record_step', default=100, type=int,
                    help='save the model every {record_step} round')

parser.add_argument('--record_res_step', default=20, type=int,
                    help='save the model every {record_res_step} round')

# === roseagg ===
parser.add_argument('--threshold', default=0.2, type=float,
                    help='similarity threshold between two model updates, >{threshold} ---> cluster')

parser.add_argument('--gradient_correction', default=0, type=int,
                    help='whether correct the gradient')

parser.add_argument('--correction_coe', default=0.1, type=float,
                    help='weight of previous gradient')

parser.add_argument('--perturbation_coe', default=0.8, type=float,
                    help='weight of random noise')

parser.add_argument('--windows', default=0, type=int,
                    help='window of previous gradient')

parser.add_argument('--cie_evaluation', default=0, type=int,
                    help='ablation: evaluate clean ingredient analysis')

# === rlr ===
parser.add_argument('--robustLR_threshold', default=10, type=int,
                    help='')


parser.add_argument('--run_name', default=None, type=str,
                    help='name of this experiment run (for wandb)')

parser.add_argument('--start_epoch', default=2001, type=int,
                    help='Load pre-trained benign model that has been trained '
                         'for start_epoch - 1 epoches, and resume from here')

parser.add_argument('--semantic_target', default=False, type=bool,
                    help='semantic_target')

parser.add_argument('--defense', default=True, type=bool,
                    help='defense')

parser.add_argument('--s_norm', default=1.0, type=float,
                    help='s_norm')

parser.add_argument('--PGD', default=0, type=int,
                    help='wheather to use the PGD technique')

parser.add_argument('--attack_num', default=40, type=int,
                    help='attack_num 10, 20, 30')

parser.add_argument('--edge_case', default=0, type=int,
                    help='edge_case or not')

parser.add_argument('--show_process', default=0, type=int)

parser.add_argument('--aggregate_all_layer', default=1, type=int)

args = parser.parse_args()
