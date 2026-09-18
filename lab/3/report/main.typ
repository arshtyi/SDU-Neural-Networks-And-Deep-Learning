#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let institute = "计算机科学与技术"
#let course = "神经网络与深度学习"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let email = link("mailto:arshtyi@foxmail.com")
#let date = datetime.today()
#let title = "实验3：全连接网络实现鸢尾花分类"

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
#let selected = results.find(result => result.name == metrics.selected_experiment)
#let percent(value) = eval(mode: "math", str(calc.round(value * 100, digits: 2)) + "%")
#let decimal(value) = eval(mode: "math", str(calc.round(value, digits: 4)))

= 实验目的：

+ 理解全连接网络、ReLU、交叉熵及反向传播，使用 PyTorch 完成鸢尾花三分类。
+ 比较原始数据、标准化和归一化对收敛过程及分类性能的影响。
+ 调整隐藏层宽度、深度和学习率，结合 Loss、Accuracy 曲线及测试指标分析结果。

= 实验软件和硬件环境：

- macos: m5, 26.6.2
- else: uv manage and `output/metrics.json`

= 实验原理和方法：

== 数据划分与预处理

使用 `load_iris()` 加载 $150$ 个样本，每个样本包含萼片长度、萼片宽度、花瓣长度、花瓣宽度四个特征，单位为厘米；标签 $0$、$1$、$2$ 分别对应 setosa、versicolor、virginica，各 $50$ 个样本。

先按 $80%$ 和 $20%$ 分层划分为 $120$ 个训练样本和 $30$ 个测试样本，每类分别有 $40$ 和 $10$ 个。为选择超参数，再从训练集中分出 $96$ 个拟合样本和 $24$ 个验证样本，每类分别有 $32$ 和 $8$ 个。两次划分均固定随机种子为 $42$。

三种预处理分别为：不缩放的原始数据（raw）；标准化（standard）；缩放至 $[0,1]$ 的 Min-Max 归一化（minmax）。后两者的公式为

$ x'_j = (x_j - mu_j) / sigma_j, quad x'_j = (x_j - m_j) / (M_j - m_j). $

其中均值 $mu_j$、标准差 $sigma_j$、最小值 $m_j$、最大值 $M_j$ 均仅由当前训练分区计算。验证阶段在 $96$ 个拟合样本上拟合缩放器；最终重训时在全部 $120$ 个训练样本上重新拟合。验证集、测试集仅调用 `transform`，不参与统计量估计。Min-Max 的 $[0,1]$ 范围针对训练数据，未见样本可能超出该范围。

== 网络与训练方法

基准网络为 $4 arrow 16 arrow 3$，即四维输入、一个含 $16$ 个神经元的隐藏层和三维输出，共 $(4+1) times 16+(16+1) times 3=131$ 个可训练参数。隐藏层使用 ReLU，引入非线性；输出层给出 logits：

$ bold(h) = "ReLU"(W_1 bold(x) + bold(b)_1), quad bold(z) = W_2 bold(h) + bold(b)_2. $

softmax 将 logits 转为类别概率，交叉熵衡量真实类别的预测概率：

$ p_c = exp(z_c) / sum_(k=0)^2 exp(z_k), quad L = -1/N sum_(i=1)^N ln p_(i,y_i). $

训练时直接将 logits 和整数标签送入 `CrossEntropyLoss`，其内部已包含等价的 log-softmax 计算，无需手动添加 softmax。预测时用 softmax 保存概率，以最大概率对应的类别作为预测结果。

所有配置均使用 Adam、批量大小 $16$、固定训练 $200$ 轮，不使用早停。每轮打乱拟合样本，依次执行梯度清零、前向传播、反向传播及参数更新。每轮结束后重新评估完整拟合集和验证集，记录平均损失与准确率，避免把最后一个批次的指标当作整轮指标。

= 实验步骤：(不要求罗列完整源代码)

== 实现与运行

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.py
```

== 对照实验与参数选择

预先设定七组配置：前三组只改变预处理；后四组以 standard 为基准，分别改变隐藏层宽度、深度或学习率。每次训练重置随机种子，批次顺序相同，相同结构的网络具有相同初始权重。

先比较 $200$ 轮后的验证准确率，同分时选择验证损失较低的配置，仍相同时保留配置顺序。确定配置后，各方案均从头在完整 $120$ 个训练样本上训练 $200$ 轮，并各测试一次，以完成指导书要求的性能对照；测试结果不参与参数选择。

#figure(
    table(
        columns: (1.2fr, 1fr, .8fr, 1fr, 1fr, 1fr),
        align: center + horizon,
        inset: 4pt,
        table.header([*配置*], [*网络结构*], [*学习率*], [*验证准确率*], [*验证损失*], [*测试准确率*]),
        ..results
            .map(result => (
                result.name,
                ((4,) + result.hidden_sizes + (3,)).map(str).join("–"),
                str(result.learning_rate),
                percent(result.selection.validation_accuracy),
                decimal(result.selection.validation_loss),
                percent(result.test.accuracy),
            ))
            .flatten(),
    ),
    caption: [七组固定配置的结果。验证指标来自 $24$ 个验证样本，测试指标来自 $120$ 个样本重训后的模型；后四组均使用标准化。],
)

验证集选定 `standard`：验证准确率为 #percent(selected.selection.validation_accuracy)，在同准确率的方案中验证损失最低，为 #decimal(selected.selection.validation_loss)。保存各配置的模型权重、网络结构、类别名及缩放参数至 `output/models/`，以便恢复完整预测流程。

= 结论分析与体会：

== 选定模型的分类结果

选定的标准化基准模型重训后，训练准确率为 #percent(selected.final_train.train_accuracy)，测试集正确 #selected.test.correct 个，共 $30$ 个，准确率为 #percent(selected.test.accuracy)，宏平均 F1 为 #decimal(selected.classification_report.at("macro avg").at("f1-score"))。

#figure(
    table(
        columns: (1.3fr, 1fr, 1fr, 1fr, .7fr),
        table.header([*类别*], [*精确率*], [*召回率*], [*F1*], [*样本数*]),
        ..metrics
            .dataset
            .class_names
            .map(name => {
                let row = selected.classification_report.at(name)
                (name, decimal(row.precision), decimal(row.recall), decimal(row.at("f1-score")), str(int(row.support)))
            })
            .flatten(),
    ),
    caption: [选定模型的测试分类指标],
)

混淆矩阵为 $mat(10, 0, 0; 0, 9, 1; 0, 0, 10)$，行是真实类别、列是预测类别，类别顺序同上表。唯一误分类样本的原始索引为 $77$（从 $0$ 开始），四个特征为 $(6.7,3.0,5.0,1.7)$，真实类别为 versicolor，被预测为 virginica。模型给这两类的概率分别约为 $0.3934$ 和 $0.6066$，说明该样本的类别判断仍有不确定性。

== 三种预处理的影响

#figure(
    image("../output/preprocessing_curves.png"),
    caption: [原始数据、标准化与 Min-Max 归一化的 Loss、Accuracy 曲线],
)

标准化在训练前期损失下降较快：第 $20$ 轮拟合集损失为 $0.0708$，原始数据和归一化分别为 $0.2140$、$0.2172$。三者最终拟合准确率均为 $97.92%$，验证准确率均为 $95.83%$。原始数据的曲线有更多局部波动，但中后期验证损失也有低于标准化的时段，不能认为标准化在所有轮次均占优。

完整训练集重训后，三者测试准确率均为 $96.67%$；测试损失依次为原始数据 $0.0757$、标准化 $0.0539$、归一化 $0.0618$。本次标准化改善了前期优化速度，且最终测试交叉熵较低，但没有提高测试准确率。准确率仅考察类别是否正确，损失还考虑预测概率，因此相同准确率不意味着相同损失。

== 网络结构与学习率的影响

#figure(
    image("../output/hyperparameter_curves.png", width: 96%),
    caption: [标准化条件下，基准网络与宽度、深度、学习率调整的对比],
)

- *隐藏层宽度：* 将 $16$ 个神经元增至 $32$ 个，拟合集最终准确率达到 $100%$，但验证准确率仍为 $95.83%$，测试准确率降至 $93.33%$。训练拟合更充分，并未带来更好的测试表现。
- *隐藏层深度：* 使用 $4 arrow 16 arrow 8 arrow 3$ 后，最终验证准确率为 $91.67%$；重训后的测试准确率仍为 $96.67%$。更深的网络增加了表示能力，但对该小数据集未体现出优势。
- *学习率：* $0.001$ 下降较慢，第 $20$ 轮拟合集损失仍为 $0.5642$，最终测试准确率为 $93.33%$；$0.05$ 前期下降很快，但后期验证损失波动更明显，最后为 $0.2744$，高于基准的 $0.0742$。本次 $0.01$ 在收敛速度与最终验证表现之间较合适。

= 就实验过程中遇到和出现的问题，你是如何解决和处理的，自拟1一3道问答题：

== 为什么网络末尾没有显式添加 softmax？

`CrossEntropyLoss` 接收原始 logits，内部计算 log-softmax 与负对数似然。若先做 softmax 再传入，会重复转换并改变损失含义。训练时直接传入 logits；保存预测概率时再调用 softmax，类别由 `argmax` 得到。

== 为什么标准化、归一化都要先划分数据再拟合？

均值、标准差和最值也是从数据中学习得到的参数。若使用全部数据计算，验证集和测试集的信息就会泄漏到训练过程。程序仅对当前训练分区调用 `fit_transform`，对待评估分区调用 `transform`，并随模型保存同一套缩放参数。

== 为什么使用验证集选参数，而且各方案还要在完整训练集上重训？

验证集用于比较方案，测试集保留作最终评估。内部验证确定配置后，从头使用全部 $120$ 个训练样本可利用原验证样本参与训练；所有方案均按同一流程重训，使三种预处理和超参数的最终测试比较保持一致。曲线和表格明确区分两个阶段，测试分数不再反过来修改配置。
