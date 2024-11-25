# ELMO

ELMo是一个深度带上下文的词表征模型，能同时建模（1）单词使用的复杂特征（例如，语法和语义）；（2）这些特征在上下文中会有何变化（如歧义等）。这些词向量从深度双向语言模型（biLM）的隐层状态中衍生出来，biLM是在大规模的语料上面Pretrain的。它们可以灵活轻松地加入到现有的模型中，并且能在很多NLP任务中显著提升现有的表现，比如问答、文本蕴含和情感分析等。听起来非常的exciting，它的原理也十分reasonable！下面就将针对论文及其PyTorch源码进行剖析，具体的资料参见文末的传送门。

ELMO 这个名称既可以代表得到词向量的模型，也可以是得出的词向量本身，就像Word2Vec、GloVe这些名称一样，都是可以代表两个含义的。

## 1. ELMo原理

之前我们一般比较常用的词嵌入的方法是诸如Word2Vec和GloVe这种，但这些词嵌入的训练方式一般都是上下文无关的，并且对于同一个词，不管它处于什么样的语境，它的词向量都是一样的，这样对于那些有歧义的词非常不友好。因此，论文就考虑到了要根据输入的句子作为上下文，来具体计算每个词的表征，提出了ELMo（Embeddings from Language Model）。它的基本思想，用大白话来说就是，还是用训练语言模型的套路，然后把语言模型中间隐含层的输出提取出来，作为这个词在当前上下文情境下的表征，简单但很有用！

## 2. ELMo整体模型结构

对于ELMo的模型结构，其实论文中并没有给出具体的图（这点对于笔者这种想象力极差的人来说很痛苦），笔者通过整合论文里面的蛛丝马迹以及PyTorch的源码，得出它大概是下面这么个东西.

<img src="./imgs/elmo.png" alt="PLMfamily" style="zoom:200%;" />

```python
BiLM(
  (token_embedder): ConvTokenEmbedder(
    (char_embeddings): Embedding(94, 50, padding_idx=85)
    (convolutions): ModuleList(
      (0): Conv1d(50, 32, kernel_size=(1,), stride=(1,))
      (1): Conv1d(50, 32, kernel_size=(2,), stride=(1,))
      (2): Conv1d(50, 64, kernel_size=(3,), stride=(1,))
      (3): Conv1d(50, 128, kernel_size=(4,), stride=(1,))
      (4): Conv1d(50, 256, kernel_size=(5,), stride=(1,))
      (5): Conv1d(50, 512, kernel_size=(6,), stride=(1,))
      (6): Conv1d(50, 1024, kernel_size=(7,), stride=(1,))
    )
    (highways): Highway(
      (layers): ModuleList(
        (0): Linear(in_features=2048, out_features=4096, bias=True)
        (1): Linear(in_features=2048, out_features=4096, bias=True)
      )
    )
    (projection): Linear(in_features=2048, out_features=512, bias=True)
  )
  (encoder): ELMoLstmEncoder(
    (forward_layers): ModuleList(
      (0): LSTM(512, 4096, batch_first=True)
      (1): LSTM(512, 4096, batch_first=True)
    )
    (backward_layers): ModuleList(
      (0): LSTM(512, 4096, batch_first=True)
      (1): LSTM(512, 4096, batch_first=True)
    )
    (forward_projections): ModuleList(
      (0): Linear(in_features=4096, out_features=512, bias=True)
      (1): Linear(in_features=4096, out_features=512, bias=True)
    )
    (backward_projections): ModuleList(
      (0): Linear(in_features=4096, out_features=512, bias=True)
      (1): Linear(in_features=4096, out_features=512, bias=True)
    )
  )
  (classifier): Linear(in_features=512, out_features=16743, bias=True)
)
```

假设输入的句子维度为 $B ∗ W ∗ C$ ，这里的 $B$ 表示batch_size，$W$ 表示num_words，即一句话中的单词数目，在一个batch中可能需要padding，$C$ 表示max_characters_per_token，即每个单词的字符数目，这里论文里面用了固定值50，不根据每个batch的不同而动态设置，$D$  表示projection_dim，即单词输入biLMs的embedding_size，或者理解为最终生成的ELMo词向量维度的1 / 2。

从图里面看，输入的句子会经过：

- Char Encode Layer： 即首先经过一个字符编码层，因为ELMo实际上是基于char的，所以它会先对每个单词中的所有char进行编码，从而得到这个单词的表示。因此经过字符编码层出来之后的维度为 $B ∗ W ∗ D $，这就是我们熟知的对于一个句子在单词级别上的编码维度。

- biLMs：随后该句子表示会经过biLMs，即双向语言模型的建模，内部其实是分开训练了两个正向和反向的语言模型，而后将其表征进行拼接，最终得到的输出维度为 $( L + 1 ) ∗ B ∗ W ∗ 2 D $，+1 实际上是加上了最初的embedding层，有点儿像residual，后面在“biLMs”部分会详细提到。

- Scalar Mixer：紧接着，得到了biLMs各个层的表征之后，会经过一个混合层，它会将前面这些层的表示进行线性融合（后面在“生成ELMo词向量”部分会进行详细说明），得出最终的ELMo向量，维度为 $B ∗ W ∗ 2 D$ 。

这里只是对ELMo模型从全局上进行的一个统观，对每个模块里面的结构还是很懵逼？没关系，下面我们逐一来进行剖析：

## 3 字符编码层

这一层即“Char Encode Layer”，它的输入维度是 B ∗ W ∗ C ，输出维度是 B ∗ W ∗ D，经查看源码，它的结构图长这样：

<img src="imgs/elmo_1.png" alt="PLMfamily" style="zoom:200%;" />

首先，输入的句子会被reshape成 $B W ∗ C$ ，因其是针对所有的char进行处理。然后会分别经过如下几个层：

- Char Embedding：这就是正常的embedding层，针对每个char进行编码，实际上所有char的词表大概是262，其中0-255是char的unicode编码，256-261这6个分别是 `<bow>`（单词的开始）、`<eow>`（单词的结束）、 `<bos>`（句子的开始）、`<eos>`（句子的结束）、`<pow>`（单词补齐符）和`<pos>`（句子补齐符）。可见词表还是比较小的，而且没有OOV的情况出现。

- Multi-Scale卷积层：这里用的是不同scale的卷积层，注意是在宽度上扩展，而不是深度上，即输入都是一样的，卷积之间的不同在于其kernel_size和channel_size的大小不同，用于捕捉不同n-grams之间的信息，这点其实是仿照 TextCNN 的模型结构。假设有m 个这样的卷积层，其kernel_size从 k1 , k2 , . . . , km，比如1,2,3,4,5,6,7这种，其channel_size从 d1 , d2 , . . . , dm，比如32,64,128,256,512,1024这种。注意：这里的卷积都是1维卷积，即只在序列长度上做卷积。与图像中的处理类似，在卷积之后，会经过MaxPooling进行池化，这里的目的主要在于经过前面卷积出的序列长度往往不一致，后期没办法进行合并，所以这里在序列维度上进行MaxPooling，其实就是取一个单词中最大的那个char的表示作为整个单词的表示。最后再经过激活层，这一步就算结束了。根据不同的channel_size的大小，这一步的输出维度分别为BW ∗ d1 , BW ∗ d2 , . . . , BW ∗ dm。

- Concat层：上一步得出的是m个不同维度的矩阵，为了方便后期处理，这里将其在最后一维上进行拼接，而后将其reshape回单词级别的维度B ∗ W ∗ (d1 + d2 + . . . + dm )。

- Highway层：Highway（参见：https://arxiv.org/abs/1505.00387 ）是仿照图像中residual的做法，在NLP领域中常有应用，看代码里面的实现，这一层实现的公式见下面：其实就是一种全连接+残差的实现方式，只不过这里还需要一个element-wise的gate矩阵对x xx和f ( A ( x ) ) f(A(x))f(A(x))进行变换。这里需要经过H HH 层这样的Highway层，输出维度仍为 B * W * (d1+d2+...+dm)。

- Linear映射层：经过前面的计算，得到的向量维度d1+d2+...+dm往往比较长，这里额外加了一层的Linear进行映射，将维度映射到D，作为词的embedding送入后续的层中，这里输出的维度为B∗W∗D。

## 4. biLMs

<img src="./imgs/elmo_2.png" alt="PLMfamily" style="zoom:200%;" />

这里的 h 表示LSTM单元的hidden_size，可能会比较大，比如D=512,h=4096这样。所以在每一层结束后还需要一个Linear层将维度从 h 映射为 D，而后再输入到下一层中。最后的输出是将每一层的所有输出以及embedding的输出，进行stack，每一层的输出里面又是对每个timestep的正向和反向的输出进行concat，因而最后的输出维度为(L+1)∗B∗W∗2D，这里的 L+1 中的 +1 就代表着那一层embedding输出，其会复制成两份，以与biLMs每层的输出维度保持一致。

## 5. 生成ELMo词向量

ELMo具有如下的优良特性：

- 上下文相关：每个单词的表示取决于使用它的整个上下文。
- 深度：单词表示组合了深度预训练神经网络的所有层。
- 基于字符：ELMo表示纯粹基于字符，然后经过CharCNN之后再作为词的表示，解决了OOV问题，而且输入的词表也很小。

## Reference

- [ELMo解读（论文 + PyTorch源码](https://blog.csdn.net/Magical_Bubble/article/details/89160032)

- 论文：https://arxiv.org/pdf/1802.05365.pdf

- 项目首页：https://allennlp.org/elmo

- 源码：https://github.com/allenai/allennlp （PyTorch，关于ELMo的部分戳这里）
  https://github.com/allenai/bilm-tf （TensorFlow）

- 多语言：https://github.com/HIT-SCIR/ELMoForManyLangs （哈工大CoNLL评测的多国语言ELMo，还有繁体中文的）
