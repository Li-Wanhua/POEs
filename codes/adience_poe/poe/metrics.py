import torch
import numpy as np
import torch.nn.functional as F


def cal_mae_acc_rank(logits, targets, is_sto=True):
    if is_sto:
        r_dim, s_dim, out_dim = logits.shape
        assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
        logits = logits.view(r_dim, s_dim, out_dim / 2, 2)
        logits = torch.argmax(logits, dim=-1)
        logits = torch.sum(logits, dim=-1)
        logits = torch.mean(logits.float(), dim=0)
        logits = logits.cpu().data.numpy()
        targets = targets.cpu().data.numpy()
        mae = sum(abs(logits - targets)) * 1.0 / len(targets)
        acc = sum(np.rint(logits) == targets) * 1.0 / len(targets)
    else:
        s_dim, out_dim = logits.shape
        assert out_dim % 2 == 0, "outdim {} wrong".format(out_dim)
        logits = logits.view(s_dim, out_dim / 2, 2)
        logits = torch.argmax(logits, dim=-1)
        logits = torch.sum(logits, dim=-1)
        logits = logits.cpu().data.numpy()
        targets = targets.cpu().data.numpy()
        mae = sum(abs(logits - targets)) * 1.0 / len(targets)
        acc = sum(np.rint(logits) == targets) * 1.0 / len(targets)
    return mae, acc


def cal_mae_acc_reg(logits, targets, is_sto=True):
    if is_sto:
        logits = logits.mean(dim=0)

    assert logits.view(-1).shape == targets.shape, "logits {}, targets {}".format(
        logits.shape, targets.shape)

    logits = logits.cpu().data.numpy().reshape(-1)
    targets = targets.cpu().data.numpy()
    mae = sum(abs(logits - targets)) * 1.0 / len(targets)
    acc = sum(np.rint(logits) == targets) * 1.0 / len(targets)

    return mae, acc


def cal_mae_acc_cls(logits, targets, is_sto=True):
    if is_sto:
        r_dim, s_dim, out_dim = logits.shape
        label_arr = torch.arange(0, out_dim).float().cuda()
        probs = F.softmax(logits, -1)
        exp = torch.sum(probs * label_arr, dim=-1)
        exp = torch.mean(exp, dim=0)
        max_a = torch.mean(probs, dim=0)
        max_data = max_a.cpu().data.numpy()
        max_data = np.argmax(max_data, axis=1)
        target_data = targets.cpu().data.numpy()
        exp_data = exp.cpu().data.numpy()
        mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    else:
        s_dim, out_dim = logits.shape
        probs = F.softmax(logits, -1)
        probs_data = probs.cpu().data.numpy()
        target_data = targets.cpu().data.numpy()
        max_data = np.argmax(probs_data, axis=1)
        label_arr = np.array(range(out_dim))
        exp_data = np.sum(probs_data * label_arr, axis=1)
        mae = sum(abs(exp_data - target_data)) * 1.0 / len(target_data)
        acc = sum(np.rint(exp_data) == target_data) * 1.0 / len(target_data)

    return mae, acc


def get_metric(main_loss_type):
    assert main_loss_type in ['cls', 'reg', 'rank'], \
        "main_loss_type not in ['cls', 'reg', 'rank'], loss type {%s}" % (
            main_loss_type)
    if main_loss_type == 'cls':
        return cal_mae_acc_cls
    elif main_loss_type == 'reg':
        return cal_mae_acc_reg
    elif main_loss_type == 'rank':
        return cal_mae_acc_rank
    else:
        raise AttributeError('main loss type: {}'.format(main_loss_type))
