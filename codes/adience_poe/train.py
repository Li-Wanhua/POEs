import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from poe import dataset_manger
import os
import time
import numpy as np
from poe import vgg
from poe.probordiloss import ProbOrdiLoss
from poe.metrics import get_metric
from poe.utils import is_fc, get_current_time, AverageMeter
from poe.options import get_args


def train_epoch(epoch):

    batch_time = AverageMeter(args.AverageMeter_MaxCount)
    loss1es = AverageMeter(args.AverageMeter_MaxCount)
    loss2es = AverageMeter(args.AverageMeter_MaxCount)
    loss3es = AverageMeter(args.AverageMeter_MaxCount)
    losses = AverageMeter(args.AverageMeter_MaxCount)
    mae = AverageMeter(args.AverageMeter_MaxCount)
    acc = AverageMeter(args.AverageMeter_MaxCount)

    rpr_model.train()

    end_time = time.time()
    for batch_idx, (inputs, targets, mh_targets) in enumerate(trainloader):

        inputs, targets, mh_targets = inputs.cuda(), targets.cuda(), mh_targets.cuda()
        inputs, targets = Variable(
            inputs, requires_grad=True), Variable(targets)

        optimizer.zero_grad()

        logit, emb, log_var = rpr_model(
            inputs, max_t=args.max_t, use_sto=args.use_sto)

        loss1, loss2, loss3, loss = criterion(
            logit, emb, log_var, targets, mh_targets, use_sto=args.use_sto)

        loss.backward()
        optimizer.step()

        batch_mae, batch_acc = cal_mae_acc(logit, targets, args.use_sto)
        loss1es.update(loss1.cpu().data.numpy())
        loss2es.update(loss2.cpu().data.numpy())
        loss3es.update(loss3.cpu().data.numpy())
        losses.update(loss.cpu().data.numpy())

        mae.update(batch_mae)
        acc.update(batch_acc)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [%d][%d/%d]  '
                  'Time %.3f (%.3f)\t'
                  'Loss1 %.3f (%.3f)\t'
                  'Loss2 %.3f (%.3f)\t'
                  'Loss3 %.3f (%.3f)\t'
                  'Loss %.3f (%.3f)\t'
                  'MAE %.3f (%.3f)\t'
                  'ACC %.3f (%.3f)' % (epoch, batch_idx, len(trainloader),
                                       batch_time.val, batch_time.avg, loss1es.val, loss1es.avg, loss2es.val, loss2es.avg, loss3es.val, loss3es.avg,
                                       losses.val, losses.avg, mae.val, mae.avg, acc.val, acc.avg))


def test_epoch(epoch):

    batch_time = AverageMeter(args.AverageMeter_MaxCount)
    loss1es = AverageMeter(args.AverageMeter_MaxCount)
    loss2es = AverageMeter(args.AverageMeter_MaxCount)
    loss3es = AverageMeter(args.AverageMeter_MaxCount)
    losses = AverageMeter(args.AverageMeter_MaxCount)
    mae_sto = AverageMeter(args.AverageMeter_MaxCount)
    mae_no_sto = AverageMeter(args.AverageMeter_MaxCount)
    acc_sto = AverageMeter(args.AverageMeter_MaxCount)
    acc_no_sto = AverageMeter(args.AverageMeter_MaxCount)

    rpr_model.eval()
    total = 0
    all_mae_sto = .0
    all_mae_no_sto = .0
    all_acc_sto = .0
    all_acc_no_sto = .0

    end_time = time.time()
    for batch_idx, (inputs, targets, mh_targets) in enumerate(testloader):
        inputs, targets, mh_targets = inputs.cuda(), targets.cuda(), mh_targets.cuda()
        inputs, targets = Variable(
            inputs, requires_grad=True), Variable(targets)

        logit, emb, log_var = rpr_model(inputs, max_t=args.max_t, use_sto=True)
        logit_no_sto, emb, log_var = rpr_model(
            inputs, max_t=args.max_t, use_sto=False)
        if args.use_sto:
            loss1, loss2, loss3, loss = criterion(
                logit, emb, log_var, targets, mh_targets, use_sto=args.use_sto)
        else:
            loss1, loss2, loss3, loss = criterion(
                logit_no_sto, emb, log_var, targets, mh_targets, use_sto=args.use_sto)

        total += targets.size(0)

        batch_mae_sto, batch_acc_sto = cal_mae_acc(logit, targets, True)
        batch_mae_no_sto, batch_acc_no_sto = cal_mae_acc(
            logit_no_sto, targets, False)

        loss1es.update(loss1.cpu().data.numpy())
        loss2es.update(loss2.cpu().data.numpy())
        loss3es.update(loss3.cpu().data.numpy())
        losses.update(loss.cpu().data.numpy())

        mae_sto.update(batch_mae_sto)
        mae_no_sto.update(batch_mae_no_sto)
        acc_sto.update(batch_acc_sto)
        acc_no_sto.update(batch_acc_no_sto)

        all_mae_sto = all_mae_sto + batch_mae_sto * targets.size(0)
        all_mae_no_sto = all_mae_no_sto + batch_mae_no_sto * targets.size(0)
        all_acc_sto = all_acc_sto + batch_acc_sto * targets.size(0)
        all_acc_no_sto = all_acc_no_sto + batch_acc_no_sto * targets.size(0)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_idx % args.print_freq == 0:
            print('Test: [%d/%d]\t'
                  'Time %.3f (%.3f)\t'
                  'Loss1 %.3f (%.3f)\t'
                  'Loss2 %.3f (%.3f)\t'
                  'Loss3 %.3f (%.3f)\t'
                  'Loss %.3f (%.3f)\t'
                  'MAE_sto %.3f (%.3f)\t'
                  'MAE_no_sto %.3f (%.3f)\t'
                  'ACC_sto %.3f (%.3f)\t'
                  'ACC_no_sto %.3f (%.3f)\t' % (batch_idx, len(testloader),
                                                batch_time.val, batch_time.avg, loss1es.val, loss1es.avg, loss2es.val, loss2es.avg, loss3es.val, loss3es.avg,
                                                losses.val, losses.avg, mae_sto.val, mae_sto.avg, mae_no_sto.val, mae_no_sto.avg, acc_sto.val, acc_sto.avg, acc_no_sto.val, acc_no_sto.avg))

    print('Test:  MAE_sto: %.3f  MAE_no_sto: %.3f ACC_sto: %.3f  ACC_no_sto: %.3f' % (all_mae_sto *
                                                                                      1.0 / total, all_mae_no_sto * 1.0 / total, all_acc_sto * 1.0 / total, all_acc_no_sto * 1.0 / total))

    return all_mae_sto * 1.0 / total, all_mae_no_sto * 1.0 / total, all_acc_sto * 1.0 / total, all_acc_no_sto * 1.0 / total


if __name__ == "__main__":
    # Training settings
    # ---------------------------------
    args = get_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ---------------------------------

    # dataset prepare
    # ---------------------------------
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomRotation(10),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainset = dataset_manger.dataset_manger(
        images_root=args.train_images_root, data_file=args.train_data_file, transforms=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = dataset_manger.dataset_manger(
        images_root=args.test_images_root, data_file=args.test_data_file, transforms=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    # --------------------------------------

    # define Model
    # ----------------------------------------
    start_epoch = 0
    rpr_model = vgg.vgg16(pretrained=True, num_output_neurons=args.num_output_neurons)
    rpr_model.cuda()
    # rpr_model = torch.nn.DataParallel(rpr_model)
    print("Model finshed")
    # ---------------------------------

    # define Optimizer
    # ----------------------------------------
    params = []
    for keys, param_value in rpr_model.named_parameters():
        if (is_fc(keys)):
            params += [{'params': [param_value], 'lr':args.fc_lr}]
        else:
            params += [{'params': [param_value], 'lr':args.lr}]

    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    lr_decay = args.lr_decay
    lr_decay_epoch = [int(i)
                      for i in args.lr_decay_epoch.split(',')] + [np.inf]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_decay_epoch, gamma=lr_decay, last_epoch=-1)
    # ---------------------------------

    # define loss
    # ----------------------------------------
    criterion = ProbOrdiLoss(distance=args.distance, alpha_coeff=args.alpha_coeff,
                             beta_coeff=args.beta_coeff, margin=args.margin, main_loss_type=args.main_loss_type)
    criterion.cuda()

    # define Metric
    # ----------------------------------------
    cal_mae_acc = get_metric(args.main_loss_type)
    # ---------------------------------

    args.logdir = os.path.join(args.logdir, args.exp_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
        print('log dir [{}] {}'.format(args.logdir, 'is created!'))

    print("start training...")
    best_mae = np.inf

    for epoch in range(start_epoch, args.max_epochs):
        print('[{}] Epoch: {} start!'.format(get_current_time(), epoch))

        if not args.test_only:
            train_epoch(epoch)
            lr_scheduler.step()
            with torch.no_grad():
                cur_mae_sto, cur_mae_no_sto, cur_acc_sto, cur_acc_no_sto = test_epoch(
                    epoch)
        else:
            with torch.no_grad():
                cur_mae_sto, cur_mae_no_sto, cur_acc_sto, cur_acc_no_sto = test_epoch(
                    epoch)
            break

        print('saving model...')
        is_best = cur_mae_no_sto < best_mae
        best_mae = min(best_mae, cur_mae_no_sto)
        if epoch % args.save_freq == 0:
            torch.save(
                {
                    'model_state_dict': rpr_model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'mae': cur_mae_no_sto,
                    'acc': cur_acc_no_sto,
                    'epoch': epoch + 1
                },
                os.path.join(args.logdir, 'checkpoint.pth')
            )
            print('save checkpoint at {}'.format(
                os.path.join(args.logdir, 'checkpoint.pth')))
        if is_best:
            torch.save(
                {
                    'model_state_dict': rpr_model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'mae': cur_mae_no_sto,
                    'acc': cur_acc_no_sto,
                    'epoch': epoch + 1
                },
                os.path.join(args.logdir, 'best.pth')
            )
            print('save best model at {}'.format(
                os.path.join(args.logdir, 'best.pth')))

        print('[{}] Epoch: {} end!'.format(get_current_time(), epoch))
