import paddle
import paddle.nn as nn
import paddle.nn.initializer as I


class RnnLm(nn.Layer):
    def __init__(self,
                 vocab_size,
                 punc_size,
                 hidden_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=0.0):
        super(RnnLm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.punc_size = punc_size

        self.embedder = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_ih_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            weight_hh_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        self.fc = nn.Linear(
            hidden_size,
            punc_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        x = inputs
        x_emb = self.embedder(x)
        x_emb = self.dropout(x_emb)

        y, (_, _) = self.lstm(x_emb)

        y = self.dropout(y)
        y = self.fc(y)
        y = paddle.reshape(y, shape=[-1,self.punc_size])
        logit = self.softmax(y)
        return y, logit



class CrossEntropyLossForLm(nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForLm, self).__init__()

    def forward(self, y, label):
        label = paddle.unsqueeze(label, axis=2)
        loss = paddle.nn.functional.cross_entropy(
            input=y, label=label, reduction='none')
        loss = paddle.squeeze(loss, axis=[2])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)
        return loss

        