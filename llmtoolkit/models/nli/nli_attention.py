import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, num_inputs, num_hiddens, flatten):
        super().__init__()
        net = []
        net.append(nn.Dropout(0.2))
        net.append(nn.Linear(num_inputs, num_hiddens))
        net.append(nn.ReLU())
        if flatten:
            net.append(nn.Flatten(start_dim=1))
        net.append(nn.Dropout(0.2))
        net.append(nn.Linear(num_hiddens, num_hiddens))
        net.append(nn.ReLU())
        if flatten:
            net.append(nn.Flatten(start_dim=1))

        self.mlp = nn.Sequential(*net)

    def forward(self, x):
        return self.mlp(x)


class Attend(nn.Module):

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = MLP(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha


class Compare(nn.Module):
    """将一个序列中的词元与与该词元软对齐的另一个序列进行比较。

    - 𝐯𝐴,𝑖=𝑔([𝐚𝑖,𝜷𝑖]),𝑖=1,…,𝑚
    - 𝐯𝐵,𝑗=𝑔([𝐛𝑗,𝜶𝑗]),𝑗=1,…,𝑛.
    """

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = MLP(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


class Aggregate(nn.Module):
    """有两组比较向量 𝐯𝐴,𝑖 （ 𝑖=1,…,𝑚 ）和 𝐯𝐵,𝑗 （ 𝑗=1,…,𝑛 ）。
    在最后一步中，我们将聚合这些信息以推断逻辑关系。我们首先求和这两组比较向量：

    - 𝐯𝐴=∑𝑖=1𝑚𝐯𝐴,𝑖,
    - 𝐯𝐵=∑𝑗=1𝑛𝐯𝐵,𝑗.
    """

    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = MLP(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat


class DecomposableAttention(nn.Module):

    def __init__(self,
                 vocab,
                 embed_size,
                 num_hiddens,
                 num_inputs_attend=100,
                 num_inputs_compare=200,
                 num_inputs_agg=400,
                 **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab, embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 有3种可能的输出：蕴涵、矛盾和中性
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        # A/B的形状：（批量大小，序列A/B的词元数，embed_size）
        # f_A/f_B的形状：（批量大小，序列A/B的词元数，num_hiddens）
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        # beta的形状：（批量大小，序列A的词元数，embed_size），
        # 意味着序列B被软对齐到序列A的每个词元(beta的第1个维度)
        # alpha的形状：（批量大小，序列B的词元数，embed_size），
        # 意味着序列A被软对齐到序列B的每个词元(alpha的第1个维度)
        beta, alpha = self.attend(A, B)
        # 将一个序列中的词元与与该词元软对齐的另一个序列进行比较。
        # V_A: （批量大小，序列B的词元数，embed_size + embed_size ）
        # V_B: （批量大小，序列B的词元数，embed_size + embed_size ）
        V_A, V_B = self.compare(A, B, beta, alpha)
        # 我们有有两组比较向量 𝐯𝐴,𝑖 （ 𝑖=1,…,𝑚 ）和 𝐯𝐵,𝑗 （ 𝑗=1,…,𝑛 ）。
        # 在最后一步中，我们将聚合这些信息以推断逻辑关系。
        # 然后将两个求和结果的连结提供给函数 ℎ （一个多层感知机），以获得逻辑关系的分类结果
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat


if __name__ == '__main__':
    X = (torch.ones(128, 50, dtype=int), torch.ones(128, 50, dtype=int))
    model = DecomposableAttention(1000, 100, 200)
    print(model)
    y = model(X)
