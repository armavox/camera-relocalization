import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, InceptionV3):
        super(PoseNet, self).__init__()
        self.base_model = InceptionV3

        self.Conv2d_1a_3x3 = InceptionV3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = InceptionV3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = InceptionV3.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = InceptionV3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = InceptionV3.Conv2d_4a_3x3
        self.Mixed_5b = InceptionV3.Mixed_5b
        self.Mixed_5c = InceptionV3.Mixed_5c
        self.Mixed_5d = InceptionV3.Mixed_5d
        self.Mixed_6a = InceptionV3.Mixed_6a
        self.Mixed_6b = InceptionV3.Mixed_6b
        self.Mixed_6c = InceptionV3.Mixed_6c
        self.Mixed_6d = InceptionV3.Mixed_6d
        self.Mixed_6e = InceptionV3.Mixed_6e
        self.Mixed_7a = InceptionV3.Mixed_7a
        self.Mixed_7b = InceptionV3.Mixed_7b
        self.Mixed_7c = InceptionV3.Mixed_7c

        # Out 2
        self.pos = nn.Linear(2048, 3, bias=False)
        self.ori = nn.Linear(2048, 4, bias=False)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos(x)
        ori = self.ori(x)

        return pos, ori


class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, base_model, fixed_weight=False, dropout_rate = 0.0):
        super(GoogleNet, self).__init__()
        self.dropout_rate = dropout_rate

        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Out 2
        self.pos = nn.Linear(2048, 3, bias=True)
        self.ori = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.base_model(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos(x)
        ori = self.ori(x)

        return pos, ori


class Loss(nn.Module):
    def __init__(self, device, beta):
        self.beta = beta
        self.device = device

    def __call__(self, pos_pred, ori_pred, pos_true, ori_true):
        pos_true = pos_true.to(self.device)
        ori_true = ori_true.to(self.device)
        ori_true = F.normalize(ori_true)
        loss_pos = pos_pred - pos_true
        loss_ori = ori_pred - ori_true
        loss = loss_pos.norm() + self.beta * loss_ori.norm()
        return loss
