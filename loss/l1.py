from torch import nn


def get_loss(param):
    return L1Loss()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x):
        l1 = self.l1(x['output'], x['gt'])
        return {'tot': l1, 'L1': l1}
