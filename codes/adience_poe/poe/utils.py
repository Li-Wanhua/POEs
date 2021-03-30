import os
import torch
from datetime import datetime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, max_count=100):
        self.reset(max_count)

    def reset(self, max_count):
        self.val = 0
        self.avg = 0
        self.data_container = []
        self.max_count = max_count

    def update(self, val):
        self.val = val
        if(len(self.data_container) < self.max_count):
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / \
                len(self.data_container)
        else:
            self.data_container.pop(0)
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / self.max_count


def is_fc(para_name):
    split_name = para_name.split('.')
    if split_name[-2] == 'final':
        return True
    else:
        return False


def display_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'], param_group['initial_lr'])


def load_model(unload_model, args):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
        print(args.save_model, 'is created!')
    if not os.path.exists(os.path.join(args.save_model, 'checkpoint.txt')):
        f = open(os.path.join(args.save_model, 'checkpoint.txt'), 'w')
        print('checkpoint', 'is created!')

    start_index = 0
    with open(os.path.join(args.save_model, 'checkpoint.txt'), 'r') as fin:
        lines = fin.readlines()
        if len(lines) > 0:
            model_path, model_index = lines[0].split()
            print('Resuming from', model_path)
            if int(model_index) == 0:
                unload_model_dict = unload_model.state_dict()

                pretrained_dict = torch.load(
                    os.path.join(args.save_model, model_path))
                pretrained_dict['emd.0.weight'] = pretrained_dict['classifier.3.weight']
                pretrained_dict['emd.0.bias'] = pretrained_dict['classifier.3.bias']
                pretrained_dict['final.weight'] = pretrained_dict['classifier.6.weight']
                pretrained_dict['final.bias'] = pretrained_dict['classifier.6.bias']

                pretrained_dict = {k: v for k, v in pretrained_dict.items() if (
                    k in unload_model_dict and pretrained_dict[k].shape == unload_model_dict[k].shape)}
                print(len(pretrained_dict))
                for dict_inx, (k, v) in enumerate(pretrained_dict.items()):
                    print(dict_inx, k, v.shape)
                unload_model_dict.update(pretrained_dict)
                unload_model.load_state_dict(unload_model_dict)
            else:
                unload_model.load_state_dict(torch.load(
                    os.path.join(args.save_model, model_path)))

            start_index = int(model_index) + 1
    return start_index


def save_model(tosave_model, epoch, args):
    model_epoch = '%04d' % (epoch)
    model_path = 'model-' + model_epoch + '.pth'
    save_path = os.path.join(args.save_model, model_path)
    torch.save(tosave_model.state_dict(), save_path)
    with open(os.path.join(args.save_model, 'checkpoint.txt'), 'w') as fin:
        fin.write(model_path + ' ' + str(epoch) + '\n')


def get_current_time():
    _now = datetime.now()
    _now = str(_now)[:-7]
    return _now
