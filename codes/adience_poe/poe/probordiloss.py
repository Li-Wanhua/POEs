import torch
import torch.nn as nn
import torch.nn.functional as F


def BhattacharyyaDistance(u1, sigma1, u2, sigma2):
    sigma_mean = (sigma1 + sigma2) / 2.0
    sigma_inv = 1.0 / (sigma_mean)
    dis1 = torch.sum(torch.pow(u1 - u2, 2) * sigma_inv, dim=1) / 8.0
    dis2 = 0.5 * (torch.sum(torch.log(sigma_mean), dim=1) -
                  0.5 * (torch.sum(torch.log(sigma1), dim=1) + torch.sum(torch.log(sigma2), dim=1)))
    return dis1 + dis2


def HellingerDistance(u1, sigma1, u2, sigma2):
    return torch.pow(1.0 - torch.exp(-BhattacharyyaDistance(u1, sigma1, u2, sigma2)), 0.5)


def WassersteinDistance(u1, sigma1, u2, sigma2):
    dis1 = torch.sum(torch.pow(u1 - u2, 2), dim=1)
    dis2 = torch.sum(torch.pow(torch.pow(sigma1, 0.5) -
                               torch.pow(sigma2, 0.5), 2), dim=1)
    return torch.pow(dis1 + dis2, 0.5)


def GeodesicDistance(u1, sigma1, u2, sigma2):
    u_dis = torch.pow(u1 - u2, 2)
    std1 = sigma1.sqrt()
    std2 = sigma2.sqrt()

    sig_dis = torch.pow(std1 - std2, 2)
    sig_sum = torch.pow(std1 + std2, 2)
    delta = torch.div(u_dis + 2 * sig_dis, u_dis + 2 * sig_sum).sqrt()
    return torch.sum(torch.pow(torch.log((1.0 + delta) / (1.0 - delta)), 2) * 2, dim=1).sqrt()


def ForwardKLDistance(u1, sigma1, u2, sigma2):
    return -0.5 * torch.sum(torch.log(sigma1) - torch.log(sigma2) - torch.div(sigma1, sigma2)
                            - torch.div(torch.pow(u1 - u2, 2), sigma2) + 1, dim=1)


def ReverseKLDistance(u2, sigma2, u1, sigma1):
    return -0.5 * torch.sum(torch.log(sigma1) - torch.log(sigma2) - torch.div(sigma1, sigma2)
                            - torch.div(torch.pow(u1 - u2, 2), sigma2) + 1, dim=1)


def JDistance(u1, sigma1, u2, sigma2):
    return ForwardKLDistance(u1, sigma1, u2, sigma2) + ForwardKLDistance(u2, sigma2, u1, sigma1)


class ProbOrdiLoss(nn.Module):
    def __init__(self, distance='Bhattacharyya', alpha_coeff=0, beta_coeff=0, margin=0, main_loss_type='cls'):
        super(ProbOrdiLoss, self).__init__()
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff
        self.margin = margin
        self.zeros = torch.zeros(1).cuda()

        assert main_loss_type in ['cls', 'reg', 'rank'], \
            "main_loss_type not in ['cls', 'reg', 'rank'], loss type {%s}" % (
                main_loss_type)
        self.main_loss_type = main_loss_type

        if distance == 'Bhattacharyya':
            self.distrance_f = BhattacharyyaDistance
        elif distance == 'Wasserstein':
            self.distrance_f = WassersteinDistance
        elif distance == 'JDistance':
            self.distrance_f = JDistance
        elif distance == 'ForwardKLDistance':
            self.distrance_f = ForwardKLDistance
        elif distance == 'HellingerDistance':
            self.distrance_f = HellingerDistance
        elif distance == 'GeodesicDistance':
            self.distrance_f = GeodesicDistance
        elif distance == 'ReverseKLDistance':
            self.distrance_f = ReverseKLDistance
        else:
            print('ERROR: this distance is not supported!')
            self.distrance_f = None

    def forward(self, logit, emb, log_var, target, mh_target=None, use_sto=True):
        class_dim = logit.shape[-1]
        sample_size = logit.shape[0]  # reparameterized with max_t samples

        if self.main_loss_type == 'cls':
            if use_sto:
                CEloss = F.cross_entropy(
                    logit.view(-1, class_dim), target.repeat(sample_size))
            else:
                CEloss = F.cross_entropy(logit.view(-1, class_dim), target)
        elif self.main_loss_type == 'reg':
            if use_sto:
                CEloss = F.mse_loss(
                    logit.view(-1), target.repeat(sample_size).to(dtype=logit.dtype))
            else:
                CEloss = F.mse_loss(
                    logit.view(-1), target.to(dtype=logit.dtype))
        elif self.main_loss_type == 'rank':
            assert len(mh_target.shape) == 2, "target shape {} wrong".format(
                mh_target.shape)
            assert class_dim % 2 == 0, "class dim {} wrong".format(class_dim)
            if use_sto:
                CEloss = F.cross_entropy(
                    logit.view(-1, 2), mh_target.repeat([sample_size, 1]).view(-1))
            else:
                CEloss = F.cross_entropy(logit.view(-1, 2), mh_target.view(-1))
        else:
            raise AttributeError(
                'main loss type: {}'.format(self.main_loss_type))

        KLLoss = torch.mean(torch.sum(torch.pow(emb, 2) +
                                      torch.exp(log_var) - log_var - 1.0, dim=1) * 0.5)

        var = torch.exp(log_var)

        batch_size = emb.shape[0]
        dims = emb.shape[1]
        target_dis = torch.abs(
            target.view(-1, 1).repeat(1, batch_size) - target.view(1, -1).repeat(batch_size, 1))
        anchor_pos = [i for i in range(batch_size)]
        second_pos = [(i + 1) % batch_size for i in anchor_pos]
        target_dis = torch.abs(target_dis - torch.abs(
            target[anchor_pos] - target[second_pos]).view(-1, 1).repeat(1, batch_size))
        offset_m = torch.eye(batch_size).cuda().to(dtype=target_dis.dtype)
        target_dis = target_dis + offset_m * 1000
        target_dis[target_dis == 0] = 700
        thrid_pos = torch.argmin(target_dis, dim=1)

        anchor_sign = torch.sign(torch.abs(
            target[anchor_pos] - target[second_pos]) - torch.abs(target[anchor_pos] - target[thrid_pos]))

        emb_dis_12 = self.distrance_f(
            emb[anchor_pos, :], var[anchor_pos, :], emb[second_pos, :], var[second_pos, :])
        emb_dis_13 = self.distrance_f(
            emb[anchor_pos, :], var[anchor_pos, :], emb[thrid_pos, :], var[thrid_pos, :])

        anchor_cons = (emb_dis_13 - emb_dis_12) * \
            anchor_sign.float() + self.margin

        loss_anchor = torch.max(self.zeros, anchor_cons) * \
            torch.abs(anchor_sign).float()
        loss_mask = (anchor_cons > 0).to(dtype=anchor_sign.dtype)
        if sum(torch.abs(anchor_sign) * loss_mask) > 0:
            triple_loss = torch.sum(loss_anchor) / \
                sum(torch.abs(anchor_sign) * loss_mask)
        else:
            triple_loss = torch.tensor(0.0).cuda()

        return CEloss, KLLoss * self.alpha_coeff, triple_loss * self.beta_coeff, CEloss + self.alpha_coeff * KLLoss + self.beta_coeff * triple_loss
