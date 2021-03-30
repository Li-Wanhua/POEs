import argparse
import torch


# Training settings
def get_args():
    parser = argparse.ArgumentParser(
        description='Probabilistic Ordinal Embedding (POE) for  age estimation for Adience dataset')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--max-epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--fc-lr', type=float, default=0.0001,
                        help='fc layer learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--AverageMeter-MaxCount', default=100, type=int,
                        help='maximum capacity for AverageMeter(default: 100)')
    parser.add_argument('--num-workers', default=2, type=int,
                        help='number of load data workers (default: 2)')
    parser.add_argument('--train-images-root', type=str, default='/home/share_data/age/CVPR19/datasets/MORPH',
                        help='images root for train dataset')
    parser.add_argument('--test-images-root', type=str, default='/home/share_data/age/CVPR19/datasets/MORPH',
                        help='images root for test dataset')
    parser.add_argument('--train-data-file', type=str, default='./data_list/ET_proto_train.txt',
                        help='data file for train dataset')
    parser.add_argument('--test-data-file', type=str, default='./data_list/ET_proto_val.txt',
                        help='data file for test dataset')
    parser.add_argument('--distance', type=str, default='JDistance',
                        help='distance metric between two gaussian distribution')
    parser.add_argument('--alpha-coeff', type=float, default=1e-5, metavar='M',
                        help='alpha_coeff (default: 0)')
    parser.add_argument('--beta-coeff', type=float, default=1e-4, metavar='M',
                        help='beta_coeff (default: 1.0)')
    parser.add_argument('--margin', type=float, default=5, metavar='M',
                        help='margin (default: 1.0)')
    parser.add_argument('--logdir', type=str, default='./log/',
                        help='where you save log.')
    parser.add_argument('--exp-name', type=str, default='exp',
                        help='name of your experiment.')
    parser.add_argument('--save-freq', default=10, type=int,
                        metavar='N', help='save checkpoint frequency (default: 10)')
    parser.add_argument('--lr-decay-epoch', type=str, default='30',
                        help='epochs at which learning rate decays. default is 30.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--no-sto', action='store_true', default=False,
                        help='not using stochastic sampling when training or testing.')
    parser.add_argument('--test-only', action='store_true',
                        default=False, help='test your model, no training loop.')
    parser.add_argument('--num-output-neurons', type=int, default=1,
                        help='number of ouput neurons of your model, note that for `reg` model we use 1; `cls` model we use `num_output_classes`; and for `rank` model we use `num_output_class` * 2.')
    parser.add_argument('--main-loss-type', type=str,
                        default='reg', help='loss type in [cls, reg, rank].')
    parser.add_argument('--max-t', type=int, default=50,
                        help='number of samples during sto.')
    parser.add_argument('--save-model', type=str, default='./Saved_Model/',
                        help='where you save model')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help="checkpoint to be loaded when testing.")

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.use_sto = True

    if args.no_sto:
        args.use_sto = False
        args.alpha_coeff = .0
        args.beta_coeff = .0
        args.margin = .0
        print("no stochastic sampling when training or testing, baseline set up")

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(str(k), str(v)))
    print('-------------- End ----------------')

    return args


if __name__ == "__main__":
    args = get_args()
