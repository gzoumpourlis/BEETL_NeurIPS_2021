import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import elu

from .modules import Expression, Ensure4d
from .functions import squeeze_final_output

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)

def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
                # nn.init.kaiming_normal_(module.weight.data)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def init_glorot(module):
    if hasattr(module, "weight"):
        nn.init.xavier_uniform_(module.weight, gain=1)
    if hasattr(module, "bias"):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def init_kaiming(module):
    if hasattr(module, "weight"):
        nn.init.kaiming_normal_(module.weight.data)
    if hasattr(module, "bias"):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class EEGNetv4_class(nn.Sequential):
    """
    EEGNet v4 model from [EEGNet4]_.

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------

    .. [EEGNet4] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2018).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        drop_prob=0.0,
        finetune=False
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_prob = drop_prob
        self.finetune = finetune
        self.use_mlp_cls = True

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        ##############################################################################

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0) # b c 0 1 --> b 1 0 c
        self.conv_temporal = nn.Conv2d(1, self.F1, (1, self.kernel_length), stride=1, bias=False,\
                                        padding=(0, self.kernel_length // 2)
                                       )
        self.bnorm_temporal = nn.BatchNorm2d(self.F1, momentum=0.1, affine=True, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.in_chans, 1),
                                                max_norm=1.0,
                                                stride=1,
                                                bias=False,
                                                groups=self.F1,
                                                padding=(0, 0),
                                                 )
        self.bnorm_1 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.1, affine=True, eps=1e-3)
        self.activation_1 = Expression(elu)
        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=self.drop_prob)
        self.conv_separable_depth = nn.Conv2d(self.F1 * self.D,
                                                self.F1 * self.D,
                                                (1, 16),
                                                stride=1,
                                                bias=False,
                                                groups=self.F1 * self.D,
                                                padding=(0, 16 // 2),
                                              )
        self.conv_separable_point = nn.Conv2d(self.F1 * self.D,
                                                self.F2,
                                                (1, 1),
                                                stride=1,
                                                bias=False,
                                                padding=(0, 0),
                                            )
        self.bnorm_2 = nn.BatchNorm2d(self.F2, momentum=0.1, affine=True, eps=1e-3)
        self.activation_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))

        # n_out_virtual_chans = out.cpu().data.numpy().shape[2]
        # n_out_time = out.cpu().data.numpy().shape[3]

        n_out_virtual_chans = 1
        n_out_time = 12

        if self.final_conv_length == "auto":
            self.final_conv_length = n_out_time

        if not self.use_mlp_cls:
            self.conv_classifier = nn.Conv2d(self.F2,
                                                self.n_classes,
                                                (n_out_virtual_chans, self.final_conv_length),
                                                bias=True
                                             )
        else:
            self.mlp_classifier = nn.Sequential(
                    nn.Linear(self.F2 * n_out_virtual_chans * self.final_conv_length, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.n_classes)
            )
        self.permute_back = Expression(_transpose_1_0) # time in third dimension (axis=2)
        self.squeeze = Expression(squeeze_final_output)

        init_kaiming(self.conv_temporal)
        init_kaiming(self.conv_spatial)
        init_kaiming(self.conv_separable_depth)
        init_kaiming(self.conv_separable_point)
        init_kaiming(self.mlp_classifier[0])
        init_kaiming(self.mlp_classifier[2])
        init_kaiming(self.mlp_classifier[4])

    def forward(self, x):

        if self.finetune:
            with torch.no_grad():
                x = self.ensuredims(x)
                x = self.dimshuffle(x)
                x = self.conv_temporal(x)
        else:
            x = self.ensuredims(x)
            x = self.dimshuffle(x)
            x = self.conv_temporal(x)
        x = self.conv_spatial(x)
        x = self.pool_1(x)
        x = self.bnorm_1(x)
        x = self.activation_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.activation_2(x)
        x = self.pool_2(x)
        # feats shape: torch.Size([10, 16, 1, 12])
        feats = x

        if not self.use_mlp_cls:
            scores = self.conv_classifier(feats)
        else:
            feats = feats.reshape(feats.shape[0], -1)
            if self.finetune:
                feats = F.normalize(feats, p=2, dim=1)
            scores = self.mlp_classifier(feats)
            scores = scores.unsqueeze(-1).unsqueeze(-1)
        scores = self.permute_back(scores)
        scores = self.squeeze(scores)

        preds = {}
        if not self.use_mlp_cls:
            preds['feats'] = feats.reshape(feats.shape[0], -1)
        else:
            preds['feats'] = feats
        preds['scores'] = scores

        return preds
