#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.3": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let institute = "计算机科学与技术"
#let course = "神经网络与深度学习"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let email = link("mailto:arshtyi@foxmail.com")
#let date = datetime.today()
#let title = "实验5：RNN-LSTM-Transformer 实现文本分类"

#set document(title: title, author: author, date: date)
#set text(font: ((name: "lato", covers: "latin-in-cjk"), "noto serif cjk sc"), size: zh(5), lang: "zh", region: "cn")
#set par(justify: true, first-line-indent: (amount: 2em, all: true))
#set page(
    paper: "a4",
    margin: (x: 35pt, y: 35pt),
    footer: align(center, context counter(page).display("- 1 -")),
)
#set heading(numbering: numbly("", "{2:1}.", "({3:1})"))
#show heading: set text(size: zh(-4))
#{
    set underline(offset: 2.5pt, extent: 2.5pt)
    show heading: it => align(center, text(tracking: .1em, size: zh(-2), it))
    heading(numbering: none, level: 1)[山东大学 #underline[#institute] 学院\ #underline[#course] 课程实验报告]
    set text(size: zh(-4))
    set table.cell(inset: .5em, align: left + horizon, stroke: 1pt)
    table(
        columns: (3fr, auto),
        [实验题目：#title], [学号：#id],
    )
    v(0em, weak: true)
    table(
        columns: (3fr, 2.5fr, 3fr),
        [日期：#date.display("[year].[month].[day]")], [班级：#class], [姓名：#author],
    )
    v(0em, weak: true)
    table(
        columns: 1fr,
        [Email：#email]
    )
}
#show raw: set text(font: ("JetBrains Mono", "Noto Serif CJK SC"))
#show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: .3em, y: 0em),
    outset: (x: 0em, y: .3em),
    radius: .2em,
)
#show: codly-init
#codly(
    languages: codly-languages,
    zebra-fill: none,
    fill: luma(90.2%),
    stroke: .5pt + rgb("bfbfbf"),
    radius: 8pt,
)
#set enum(numbering: numbly("{1:1})", "{2:a}."))
#set list(indent: 10pt, marker: sym.bullet.tri)
#set table(inset: 5pt, align: center + horizon, stroke: .5pt + luma(160))
#show table.cell.where(y: 0): set text(weight: "bold")

#let in-block(body) = {
    let is-level-1-heading(it) = (
        it.func() == heading
            and (
                it.at("level", default: none) == 1
                    or (it.at("offset", default: none) + it.at("depth", default: none) == 1)
            )
    )

    let text-block(it) = {
        v(0em, weak: true)
        block(
            width: 100%,
            inset: (x: 4pt, y: 1em),
            stroke: 1pt,
            breakable: true,
            it,
        )
    }

    let children = body.at("children", default: (body,))
    let content = ()
    let buf = ()

    for child in children {
        if is-level-1-heading(child) {
            if buf.len() > 0 {
                content.push(text-block(buf.join()))
                buf = ()
            }
            buf.push(child)
        } else if buf.len() > 0 {
            buf.push(child)
        } else {
            content.push(child)
        }
    }
    if buf.len() > 0 {
        content.push(text-block(buf.join()))
    }
    content.join()
}
#show: in-block

#let metrics = json("../output/metrics.json")
#let results = metrics.experiments
#let selected = metrics.selected_experiments
#let get-result(name) = results.find(result => result.name == name)
#let rnn = get-result(selected.rnn)
#let lstm = get-result(selected.lstm)
#let transformer = get-result(selected.transformer)
#let dataset = metrics.dataset
#let long-text = metrics.long_text
#let percent(value, digits: 2) = eval(mode: "math", str(calc.round(value * 100, digits: digits)) + "%")
#let decimal(value, digits: 4) = eval(mode: "math", str(calc.round(value, digits: digits)))
#let model-name(architecture) = (rnn: "BiRNN", lstm: "BiLSTM", transformer: "Transformer").at(architecture)

= 实验目的：

+ 理解普通循环神经网络、LSTM 门控机制和 Transformer 自注意力机制，完成 AG_NEWS 英文新闻四分类。
+ 使用 PyTorch 与 torchtext，实现分词、训练词表、序列编码、双向循环网络、位置编码、训练、验证及独立测试。
+ 比较三类模型的准确率、交叉熵、训练时间与学习率影响，目标是三类模型的测试准确率均达到 $90%$ 以上。
+ 对指导书中的完整新闻段落进行分类，展示原文、真实类别、模型预测及概率，并保存可复用的模型与词表。

= 实验软件和硬件环境：

- macos: m5
- else: uv manage and `output/metrics.json`

= 实验原理和方法：

== 数据集、划分与统计

通过 `torchtext.datasets.AG_NEWS` 下载数据。每条样本由类别编号、标题和新闻摘要组成，加载器以空格连接标题与摘要。四类依次为 World（国际新闻）、Sports（体育）、Business（商业）、Sci/Tech（科技）；原始标签 $1$ 至 $4$ 转换为交叉熵所需的 $0$ 至 $3$。

从官方 $120000$ 条训练数据中，按类别分层划分 $108000$ 条拟合数据和 $12000$ 条验证数据；随机种子为 $42$。官方 $7600$ 条测试数据独立保留，不参与词表构建、训练、学习率选择或最优轮次选择。

#figure(
    table(
        columns: (1.1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        table.header([数据子集], [样本数], [每类数量], [平均 token], [95% 分位], [最大长度]),
        ..("fit", "validation", "test")
            .map(split => {
                let item = dataset.splits.at(split)
                (
                    (fit: "拟合", validation: "验证", test: "测试").at(split),
                    str(item.size),
                    str(item.class_counts.first()),
                    decimal(item.mean_tokens, digits: 2),
                    str(item.p95_tokens),
                    str(item.max_tokens),
                )
            })
            .flatten(),
    ),
    caption: [分词后的数据统计；每个子集的四类样本数相同],
)

下载文件的校验值：

#table(
    columns: (1fr, 3fr),
    table.header([文件], [MD5]),
    [train.csv], text(size: 9pt, dataset.md5.at("train.csv")),
    [test.csv], text(size: 9pt, dataset.md5.at("test.csv")),
)

== 分词、词表与序列表示

使用 torchtext 的 `basic_english` 分词器，将文本小写并按规则分离标点。词表仅统计拟合子集，最低词频为 $2$，最大词表大小为 $30000$（包含特殊符号），实际大小为 #dataset.vocabulary_size。`<pad>` 编号为 $0$，`<unk>` 编号为 $1$；未知词映射到 `<unk>`，推理时不扩充词表。

训练和标准测试把序列截断到最多 $128$ 个 token，右侧补零。拟合、验证、测试子集中被截断的样本分别为 #dataset.splits.fit.truncated_count、#dataset.splits.validation.truncated_count 和 #dataset.splits.test.truncated_count 条，拟合截断比例为 #percent(dataset.splits.fit.truncated_count / dataset.splits.fit.size)。每批按实际最大长度裁去多余填充，训练中使用长度分桶降低填充计算；每轮仍完整遍历所有拟合样本，批次次序随机打乱。

三类网络均使用随机初始化、可训练的 $30000 times 128$ 词嵌入矩阵，不使用外部预训练词向量。`padding_idx=0` 使填充词向量保持为零，但仅将词向量设零还不足以排除循环状态、注意力及池化中的填充影响，因此还需分别处理掩码。

== 双向普通 RNN

普通 RNN 在每个时间步更新状态：

$ h_t = tanh(W_x x_t + W_h h_(t-1) + b). $

正向网络按词序处理，反向网络按相反词序处理；把两者在同一位置的输出拼接，得到 $H_t=[h_t^"forward"; h_t^"backward"]$。实现使用单层 `nn.RNN`、`bidirectional=True`、每方向 $64$ 个隐藏单元，所以每个有效位置输出 $128$ 维表示。

重复的状态变换可能使长距离梯度衰减或放大。训练时把整体梯度范数裁剪到 $1.0$，限制过大的更新；但梯度裁剪并不能消除梯度消失。为避免只依赖最终状态，分类器聚合所有有效位置的平均值与最大值。

== 双向 LSTM

LSTM 引入遗忘门、输入门、输出门以及细胞状态：

$
    f_t = sigma(W_f [h_(t-1); x_t] + b_f), quad
    i_t = sigma(W_i [h_(t-1); x_t] + b_i),
$
$
    tilde(c)_t = tanh(W_c [h_(t-1); x_t] + b_c), quad
    c_t = f_t dot c_(t-1) + i_t dot tilde(c)_t,
$
$
    o_t = sigma(W_o [h_(t-1); x_t] + b_o), quad
    h_t = o_t dot tanh(c_t).
$

其中 $dot$ 表示逐元素相乘。细胞状态提供受门控的加性传播路径，能缓解长距离梯度衰减，但不能保证任意长度上的信息都被保留。实现使用单层双向 `nn.LSTM`，每方向隐藏维度同为 $64$，与 RNN 保持相同输出宽度与池化方式。

两类循环网络都用 `pack_padded_sequence` 按真实长度打包，前向传播后用 `pad_packed_sequence` 还原位置。这样反向分支不会先读取一串填充值，文本表示不随同批其他样本长度改变。

== Transformer 与位置编码

Transformer 输入为词向量的缩放值与正弦位置编码之和：

$ z_t = sqrt(d) x_t + "PE"_t, quad d=128, $
$
    "PE"_(t,2i) = sin(t / 10000^(2i/d)), quad
    "PE"_(t,2i+1) = cos(t / 10000^(2i/d)).
$

每个注意力头计算

$ "Attention"(Q,K,V) = "softmax"((Q K^T)/sqrt(d_k) + M)V. $

其中 $M$ 屏蔽填充键位置。实现采用 $2$ 层 Transformer Encoder、$4$ 个注意力头、$256$ 维前馈层、GELU、残差连接和层归一化；每头维度为 $32$。分类允许双向观察整个输入，因此使用填充掩码而不使用因果掩码。没有位置编码时，自注意力无法独立表达词序信息。

== 池化、分类与损失

设有效 token 数为 $L$、编码输出为 $H_t$，则统一分类头为

$ u = [1/L sum_(t=1)^L H_t; max_(1 <= t <= L) H_t], quad z = W u+b. $

最大值在每个特征维度上分别取。平均池化显式乘有效位置掩码，最大池化把无效位置设为负无穷；三类网络的池化拼接结果均为 $256$ 维。词嵌入输出与分类头输入使用 dropout，概率为 $0.3$。

网络输出四维 logits。交叉熵与准确率分别为

$ L_"CE" = -1/N sum_(i=1)^N ln(exp(z_(i,y_i)) / (sum_(c=0)^3 exp(z_(i,c)))), $
$ "Accuracy" = 1/N sum_(i=1)^N bold(1)[arg max_c z_(i,c) = y_i]. $

`CrossEntropyLoss` 直接接收 logits；只在导出类别概率时使用 softmax。

== 实验控制与模型选择

三类模型共享同一划分、词表、最大长度、词嵌入维度、批量大小 $128$ 和 $8$ 轮训练预算。优化器为 AdamW，权重衰减为 $0.01$；学习率按余弦调度降低到初值的 $10%$。每次实验重设随机种子，使用相同的批次随机生成器。

#table(
    columns: (1.7fr, 1.25fr, 1fr, 1fr, 1fr),
    table.header([配置], [结构], [初始学习率], [层数], [参数量]),
    ..results
        .map(result => (
            text(size: 9pt, result.name),
            model-name(result.config.architecture),
            str(result.config.learning_rate),
            str(result.config.layers),
            str(result.parameters),
        ))
        .flatten(),
)

每轮保存验证准确率最高的权重，同准确率时取验证损失更小者，完全相同则保留更早轮次。全部训练完成后，再按同一规则在每类网络的两档学习率中选定代表模型，然后评估各配置的独立测试结果。报告中的测试比较不会改变已选定的模型；不合并验证集重训。

训练指标是启用 dropout 时每个批次的在线结果，按实际样本数加权；验证和测试指标在 `eval()` 与 `inference_mode()` 下计算。两种口径不同，不能把某一轮训练准确率低于验证准确率直接解释为数据泄漏。

= 实验步骤：(不要求罗列完整源代码)

== 执行与产物

#raw(read("../main.py"), block: true, lang: "python")

```sh
uv sync
uv run main.py
```
= 结论分析与体会：

== 测试准确率与损失

#figure(
    table(
        columns: (1.7fr, .7fr, 1fr, 1fr, 1fr, 1fr),
        table.header([配置], [最优轮], [验证准确率], [测试准确率], [测试损失], [宏平均 F1]),
        ..results
            .map(result => (
                text(size: 9pt, result.name),
                str(result.best.epoch),
                percent(result.best.validation_accuracy),
                percent(result.test.accuracy),
                decimal(result.test.loss),
                decimal(result.classification_report.at("macro avg").at("f1-score")),
            ))
            .flatten(),
    ),
    caption: [每个配置均恢复自身验证集最优权重后，在相同的 7600 条测试新闻上评估],
)

验证集选定 #raw(rnn.name)、#raw(lstm.name) 与 #raw(transformer.name)。它们分别正确分类 #rnn.test.correct、#lstm.test.correct 和 #transformer.test.correct 条测试新闻，准确率为 *#percent(rnn.test.accuracy)*、*#percent(lstm.test.accuracy)* 和 *#percent(transformer.test.accuracy)*。#if metrics.requirement_met [三类模型均达到测试准确率 $90%$ 以上的要求。] else [本次结果尚未使三类模型均达到测试准确率 $90%$，应如实保留结果并继续依据验证集改进。]

#figure(
    image("../output/confusion_matrices.png"),
    caption: [三类选定模型的混淆矩阵；行是真实类别，列是预测类别，每类均有 1900 条样本],
)

#figure(
    table(
        columns: (1.25fr, 1fr, 1fr, 1fr, 1fr),
        table.header([模型 / 类别], [精确率], [召回率], [F1], [样本数]),
        ..(rnn, lstm, transformer)
            .map(result => {
                let rows = dataset
                    .labels
                    .map(label => {
                        let row = result.classification_report.at(label)
                        (
                            label,
                            decimal(row.precision),
                            decimal(row.recall),
                            decimal(row.at("f1-score")),
                            str(int(row.support)),
                        )
                    })
                    .flatten()
                (
                    table.cell(colspan: 5, fill: luma(240))[#model-name(result.config.architecture)（#result.name）],
                    ..rows,
                )
            })
            .flatten(),
    ),
    caption: [选定模型的逐类分类指标],
)

#for result in (rnn, lstm, transformer) {
    let pair = result.top_confusions.first()
    par[*#model-name(result.config.architecture)* 最多的单向混淆为真实 #pair.true 被预测为 #pair.predicted，共 #pair.count 条，占该模型全部 #str(7600 - result.test.correct) 条错误的 #percent(pair.count / (7600 - result.test.correct))。]
}

Business 与 Sci/Tech 的新闻可能同时包含公司、产品、市场和技术词汇，World 与 Business 也可能共享能源、贸易等主题。混淆矩阵提供了比单一准确率更具体的错误分布；这些语义联系可帮助提出误差解释，但仅凭汇总矩阵不能确定某条新闻误判的原因。

== 训练时间与模型规模

#figure(
    table(
        columns: (1.7fr, 1fr, 1fr, 1fr, 1fr),
        table.header([配置], [参数量], [训练总秒数], [平均秒 / 轮], [墙钟秒数]),
        ..results
            .map(result => (
                text(size: 9pt, result.name),
                str(result.parameters),
                decimal(result.training_seconds, digits: 2),
                decimal(result.training_seconds / result.config.epochs, digits: 2),
                decimal(result.wall_seconds, digits: 2),
            ))
            .flatten(),
    ),
    caption: [CPU、2 线程、每配置 8 轮；训练总时间不含每轮验证与文件写入],
)

选定 BiRNN、BiLSTM、Transformer 的训练时间分别为 #decimal(rnn.training_seconds, digits: 2)、#decimal(lstm.training_seconds, digits: 2)、#decimal(transformer.training_seconds, digits: 2) 秒。词嵌入占 $30000 times 128=3840000$ 个参数，是三类网络的共同主要部分；网络总参数量相近不代表计算耗时相同。LSTM 每步计算多个门，Transformer 需计算注意力和前馈层，实际速度还受序列长度、填充比例、实现及设备影响。

自注意力允许序列位置间的并行计算，但并不保证在 CPU、小批量、短新闻任务中一定更快。

== 学习曲线与学习率对照

#figure(
    image("../output/learning_curves.png"),
    caption: [相同训练预算下的学习率对照；实线为验证指标，虚线为训练在线指标，上排准确率，下排交叉熵],
)

#for architecture in ("rnn", "lstm", "transformer") {
    let base = get-result(architecture)
    let other = get-result(architecture + "_lr")
    let chosen = get-result(selected.at(architecture))
    par[
        *#model-name(architecture)：* 学习率从 #base.config.learning_rate 调整为 #other.config.learning_rate 后，最佳验证准确率从 #percent(base.best.validation_accuracy) 变为 #percent(other.best.validation_accuracy)，变化为 #decimal((other.best.validation_accuracy - base.best.validation_accuracy) * 100, digits: 3) 个百分点。两者第一轮训练损失分别为 #decimal(base.history.first().train_loss) 与 #decimal(other.history.first().train_loss)。验证集最终选择 #raw(chosen.name)，该配置最后一轮在线训练准确率为 #percent(chosen.last.train_accuracy)、验证准确率为 #percent(chosen.last.validation_accuracy)，最优权重来自第 #chosen.best.epoch 轮。
    ]
}

== 指导书长文本的真实结果与预测结果

文本讨论 Jon Rahm 在 PGA Tour 高尔夫赛事中的表现，因此人工标注真实类别为 *Sports（体育，模型标签 1，原始 AG_NEWS 标签 2）*。

#block(
    fill: luma(96%),
    inset: 10pt,
    radius: 3pt,
    text(size: 10pt, lang: "en", long-text.text),
)

文本经同一分词器得到 #long-text.predictions.first().token_count 个 token，划分为 #long-text.predictions.first().chunk_count 个连续片段，长度依次为 #long-text.predictions.first().chunk_lengths.map(str).join("、")，所有片段均参与预测。各片段先独立计算四类 softmax 概率，再按有效 token 数加权平均：

$ p(c | x) = (sum_(j=1)^m L_j p(c | x_j)) / (sum_(j=1)^m L_j). $

本例只有一个片段，等价于对原文完整推理。对超过上限的其他文本，该方式可以覆盖全文；但它不会建立跨片段的注意力或循环状态，也不能保证能处理跨段否定与主题转折。

#figure(
    table(
        columns: (1.65fr, 1fr, 1fr, 1fr, .8fr),
        table.header([模型], [真实类别], [预测类别], [预测概率], [是否正确]),
        ..long-text
            .predictions
            .map(prediction => (
                text(size: 9pt, prediction.model),
                long-text.true_class,
                prediction.predicted_class,
                percent(prediction.confidence, digits: 4),
                if prediction.correct [正确] else [错误],
            ))
            .flatten(),
    ),
    caption: [指导书完整长文本的真实标签与模型实际预测],
)

#table(
    columns: (1.65fr, 1fr, 1fr, 1fr, 1fr),
    table.header([模型], [World], [Sports], [Business], [Sci/Tech]),
    ..long-text
        .predictions
        .map(prediction => (
            text(size: 9pt, prediction.model),
            ..dataset.labels.map(label => percent(prediction.probabilities.at(label), digits: 4)),
        ))
        .flatten(),
)

== 实验体会

在这组以短新闻为主的数据上，双向循环表示与带掩码的池化已能提取有效主题信息；模型更复杂不必然带来更高准确率或更短训练时间。完整文本演示则补充了汇总指标之外的实际使用场景，也暴露出词表外词、输入长度上限以及分段聚合的局限。

= 就实验过程中遇到和出现的问题，你是如何解决和处理的，自拟1一3道问答题：

== 为什么只把 padding 的词向量设为零还不够？

循环网络即使接收零输入，也可能因偏置与历史状态继续更新；双向网络的反向分支还会先经过右侧填充。Transformer 的填充位置也会影响注意力归一化，而最大池化可能错误选中无效位置。分别采用打包序列、注意力填充掩码与有效位置池化，并检查追加 padding 前后 logits 一致、单独推理与同批推理一致，才能验证填充被正确排除。

== 为什么最优模型不按测试准确率选择，训练曲线又可能低于验证曲线？

若用测试集选择学习率或轮次，测试集就参与调参，不能继续作为独立泛化评估。这里先固定验证最优配置，再输出测试结果。训练曲线记录参数持续更新、dropout 开启时的在线统计，验证曲线使用轮末参数且关闭 dropout，二者口径不同；因此应结合验证趋势与独立测试判断模型，而不是直接比较两条曲线的高低。
