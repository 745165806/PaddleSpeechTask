import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# ??from paddle.autograd import Variable


class FocalLoss(nn.Layer):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        # print('input')
        # print(input.shape)
        # print(target.shape)
        
        if input.dim()>2:
            input = paddle.reshape(input, shape=[input.size(0),input.size(1),-1])  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = paddle.reshape(input, shape=[-1,input.size(2)])   # N,H*W,C => N*H*W,C
        target = paddle.reshape(target, shape=[-1])

        logpt = F.log_softmax(input)
        # print('logpt')
        # print(logpt.shape)
        # print(logpt)

        # get true class column from each row
        all_rows = paddle.arange(len(input))
        # print(target)
        log_pt = logpt.numpy()[all_rows.numpy(), target.numpy()]

        pt = paddle.to_tensor(log_pt,dtype='float64').exp()
        ce = F.cross_entropy(input, target,reduction='none')
        # print('ce')
        # print(ce.shape)


        
        loss = (1-pt)**self.gamma * ce
        # print('ce:%f'%ce.mean())
        # print('fl:%f'%loss.mean())
        if self.size_average: return loss.mean()
        else: return loss.sum()