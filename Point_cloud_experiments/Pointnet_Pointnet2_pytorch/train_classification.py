"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from e3nn import o3


from pathlib import Path
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int,
                        choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--scaling_data_aug_interval', type=float, nargs='+', default=[0.8, 1.25],
                        help='Randomly scale in which interval during training?')
    parser.add_argument('--translation_data_aug_dist', type=float, default=0.1,
                        help='Randomly translate in which interval during training?')

    parser.add_argument('--random_rotations', action='store_true', help='Randomly rotate data during training.')
    parser.add_argument('--pca_align_training', action='store_true', help='Use PCA alignment during training.')
    parser.add_argument('--pca_align_testing', action='store_true', help='Use PCA alignment during testing.')

    parser.add_argument('--center_align_training', action='store_true', help='Use centering during training.')
    parser.add_argument('--center_align_testing', action='store_true', help='Use centering during testing.')

    parser.add_argument('--unscale_training', action='store_true', help='Use unscaling during training.')
    parser.add_argument('--unscale_testing', action='store_true', help='Use unscaling during testing.')

    parser.add_argument('--reset_training', action='store_true', help='Reset and train a new model from scratch.')
    parser.add_argument('--dryrun', action='store_true', help='Run all loops only once to test everything.')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def pca_align(positions):
    X = positions - positions.mean(dim=1, keepdim=True)
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    d = torch.sign(U[:, 0, :])  # get the first significant sign of u # S is ordered
    D = torch.diag_embed(d.float())

    # repeated_eigenvalues = (torch.isclose(S[:, 0], S[:, 1]) | torch.isclose(S[:, 1], S[:, 0])).nonzero()
    # if len(repeated_eigenvalues) > 0:
    #     print(repeated_eigenvalues)

    return X.matmul(V.transpose(1, 2).matmul(D))


def test(model, loader, num_class=40, device=torch.device('cpu'), dryrun=False):
    classifier = model.eval()

    mean_correct_avg = []
    class_acc_avg = np.zeros((num_class, 3))

    for j, data in enumerate(loader):
        points, target = data
        points, target = points.to(device=device), target.to(device=device)

        # Unscale:
        if args.unscale_training:
            # Set scale as average distance to origin.
            scale_estimate = points.norm(dim=2, keepdim=True).mean(dim=1, keepdim=True)
            points /= scale_estimate

        # Subtract mean here
        if args.center_align_training:
            # Set location estimate to average distance to origin.
            location_estimate = points.norm(dim=2, keepdim=True).mean(dim=1, keepdim=True)
            points -= location_estimate

        if args.pca_align_testing:  # why is args even a valid variable in this scope??
            points = pca_align(points)
        pred, _ = classifier(points.transpose(2, 1))
        pred_choice = pred.argmax(dim=1)
        predictions_avg_case = (pred_choice == target).float()

        # Process average case predictions
        for cls in np.unique(target.cpu()):
            classacc = predictions_avg_case[target == cls].cpu().sum()
            class_acc_avg[cls, 0] += classacc.item() / \
                float(predictions_avg_case[target == cls].shape[0])
            class_acc_avg[cls, 1] += 1
        mean_correct_avg.append(predictions_avg_case.sum().cpu() / float(points.size()[0]))

        if dryrun:
            break

    class_acc_avg[:, 2] = class_acc_avg[:, 0] / class_acc_avg[:, 1]
    avg_class_acc = np.mean(class_acc_avg[:, 2])
    avg_instance_acc = np.mean(mean_correct_avg)

    return avg_instance_acc, avg_class_acc


def _apply_rotation(points, theta, gamma, device=torch.device('cpu')):
    rot_xy = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                          device=device, dtype=torch.float)
    rot_yz = torch.tensor([[np.cos(gamma), -np.sin(gamma)], [np.sin(gamma), np.cos(gamma)]],
                          device=device, dtype=torch.float)
    # Rotate points
    points[:, :, 0:2] = points[:, :, 0:2].matmul(rot_xy)
    points[:, :, 1:3] = points[:, :, 1:3].matmul(rot_yz)

    # Rotate normals
    if points.shape[2] == 6:
        points[:, :, 0:2] = points[:, :, 3:5].matmul(rot_xy)
        points[:, :, 1:3] = points[:, :, 4:6].matmul(rot_yz)
    return points


def test_invariance(model, loader, num_class=40, device=torch.device('cpu'), invariance_type='scaling', dryrun=False):
    """Can test invariance to scaling, invariance to translation and invariance to rotations."""
    model.eval()

    if invariance_type == 'scaling':
        invariances = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10, 100, 1000]
    elif invariance_type == 'translation':
        shift_range = [-10.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 10.0]
        invariances = [[x, y, z] for x in shift_range for y in shift_range for z in shift_range]
    elif invariance_type == 'rotation':
        thetas = np.linspace(0, 2 * np.pi, num=16)
        gammas = np.linspace(0, 2 * np.pi, num=16)
        invariances = [[theta, gamma] for theta in thetas for gamma in gammas]
    else:
        raise ValueError('Invalid invariance_type given.')

    mean_correct_avg = []
    class_acc_avg = np.zeros((num_class, 3))
    mean_correct_worst = []
    class_acc_worst = np.zeros((num_class, 3))

    for j, data in enumerate(loader):
        points, target = data
        points_clean, target = points.to(device=device), target.to(device=device)

        # Rotate points by rotation
        predictions_worst_case = torch.ones_like(target, dtype=torch.bool)
        predictions_avg_case = torch.zeros_like(target, dtype=torch.float)
        num_trials = 0
        for invariance in invariances:
            num_trials += 1
            points = points_clean.clone()
            if invariance_type == 'scaling':
                points *= invariance
            elif invariance_type == 'translation':
                points += torch.as_tensor(invariance, device=device)[None, None, :]
            elif invariance_type == 'rotation':
                _apply_rotation(points, theta=invariance[0], gamma=invariance[1], device=device)

            # Unscale:
            if args.unscale_testing:
                # Set scale as average distance to origin.
                scale_estimate = points.norm(dim=2, keepdim=True).mean(dim=1, keepdim=True)
                points /= scale_estimate

            # Subtract mean here
            if args.center_align_testing:
                # Set location estimate to average distance to origin.
                location_estimate = points.mean(dim=1, keepdim=True)
                points -= location_estimate

            # Align to PCA direction
            if args.pca_align_testing:  # why is args even a valid variable in this scope??
                points = pca_align(points)
            pred, _ = model(points.transpose(2, 1))
            pred_choice = pred.argmax(dim=1)

            predictions_worst_case = predictions_worst_case & (pred_choice == target)
            predictions_avg_case += (pred_choice == target).float()

        # Process average case predictions
        predictions_avg_case = predictions_avg_case / num_trials
        for cls in np.unique(target.cpu()):
            classacc = predictions_avg_case[target == cls].cpu().sum()
            class_acc_avg[cls, 0] += classacc.item() / \
                float(predictions_avg_case[target == cls].shape[0])
            class_acc_avg[cls, 1] += 1
        mean_correct_avg.append(predictions_avg_case.sum().cpu() / float(points.size()[0]))
        # Process worst-case predictions if any
        for cls in np.unique(target.cpu()):
            classacc = predictions_worst_case[target == cls].cpu().sum()
            class_acc_worst[cls, 0] += classacc.item() / \
                float(predictions_worst_case[target == cls].shape[0])
            class_acc_worst[cls, 1] += 1
        correct = predictions_worst_case.eq(target.long().data).cpu().sum()
        mean_correct_worst.append(predictions_worst_case.sum().cpu() / float(points.size()[0]))
        if dryrun:
            break

    class_acc_avg[:, 2] = class_acc_avg[:, 0] / class_acc_avg[:, 1]
    avg_class_acc = np.mean(class_acc_avg[:, 2])
    avg_instance_acc = np.mean(mean_correct_avg)

    class_acc_worst[:, 2] = class_acc_worst[:, 0] / class_acc_worst[:, 1]
    worst_class_acc = np.mean(class_acc_worst[:, 2])
    worst_instance_acc = np.mean(mean_correct_worst)

    return avg_instance_acc, avg_class_acc, worst_instance_acc, worst_class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # dont
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_name = (f'{"ROT" if args.random_rotations else ""}_S{args.scaling_data_aug_interval}_T{args.translation_data_aug_dist}'
                f'PCA_{"Train" if args.pca_align_training else ""}{"Test" if args.pca_align_testing else ""}_'
                f'TL_{"Train" if args.center_align_training else ""}{"Test" if args.center_align_testing else ""}_'
                f'UC_{"Train" if args.unscale_training else ""}{"Test" if args.unscale_testing else ""}')

    exp_dir = exp_dir.joinpath(exp_name)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args,
                                       split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args,
                                      split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    print(model)
    print(classifier)
    classifier = classifier.to(device=device)
    criterion = criterion.to(device=device)

    try:
        assert not args.restart_training
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:  # noqa
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in enumerate(trainDataLoader, 0):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3], *args.scaling_data_aug_interval)
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], args.translation_data_aug_dist)
            points = torch.Tensor(points)

            points, target = points.to(device=device), target.to(device=device)

            if args.random_rotations:
                with torch.no_grad():
                    R = o3.rand_matrix(len(points)).to(device=device)
                points[:, :, :3] = points[:, :, :3].matmul(R)
                if args.use_normals:
                    points[:, :, 3:] = points[:, :, 3:].matmul(R)

            # Unscale:
            if args.unscale_training:
                # Set scale as average distance to origin.
                scale_estimate = points.norm(dim=2, keepdim=True).mean(dim=1, keepdim=True)
                points /= scale_estimate

            # Subtract mean here
            if args.center_align_training:
                # Set location estimate to average distance to origin.
                location_estimate = points.mean(dim=1, keepdim=True)
                points -= location_estimate

            # Align to PCA direction
            if args.pca_align_training:
                points = pca_align(points)
            pred, trans_feat = classifier(points.transpose(2, 1))
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            if args.dryrun:
                break

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier, testDataLoader, num_class=num_class,
                                           device=device, dryrun=args.dryrun)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' %
                       (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
            if args.dryrun:
                break

    log_string('End of training... ')
    for inv in ['rotation', 'translation', 'scaling']:
        log_string(f'Testing {inv} now ...')
        with torch.no_grad():
            instance_acc, class_acc, worst_case_instance_acc, worst_case_class_acc = test_invariance(classifier, testDataLoader,
                                                                                                     num_class=40,
                                                                                                     device=device,
                                                                                                     invariance_type=inv,
                                                                                                     dryrun=args.dryrun)
        log_string(f'{inv} Worst-case Test Instance Accuracy: {worst_case_instance_acc} - '
                   f'{inv} Average-case Test Instance Accuracy: {instance_acc}')
        log_string(f'{inv} Worst-case Test Class Accuracy: {worst_case_class_acc} - '
                   f'{inv} Average-case Test Class Accuracy: {class_acc}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
