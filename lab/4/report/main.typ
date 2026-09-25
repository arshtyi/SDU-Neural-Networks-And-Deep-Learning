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
#let title = "实验4：LeNet 实现 MNIST 图像分类"

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
#let get-result(name) = results.find(result => result.name == name)
#let mlp = get-result(metrics.selected_experiments.mlp)
#let lenet = get-result(metrics.selected_experiments.lenet)
#let baseline = get-result("lenet")
#let short = get-result("lenet_short")
#let small = get-result("lenet_lr_small")
#let large-batch = get-result("lenet_batch256")
#let c3 = metrics.c3_verification
#let history = csv("../output/history.csv", row-type: dictionary)
#let first-loss(name) = float(history.find(row => row.experiment == name and row.epoch == "1").train_loss)
#let percent(value) = eval(mode: "math", str(calc.round(value * 100, digits: 2)) + "%")
#let decimal(value) = eval(mode: "math", str(calc.round(value, digits: 4)))

= 实验目的：

+ 理解卷积、参数共享、池化与非线性激活的作用，实现 C3 非对称连接的 LeNet 手写数字分类。
+ 在 MNIST 上训练全连接网络和 LeNet，完成训练、验证、最优模型保存及独立测试，两类网络测试准确率均超过 $97%$。
+ 对比网络深度、训练轮数、学习率、批量大小对结果的影响；可视化首层及末层卷积核和特征图，分析 CNN 的特征提取过程。

= 实验软件和硬件环境：

- macos: m5, 26.6.2
- else: uv manage and `output/metrics.json`

= 实验原理和方法：

== 数据集与预处理

MNIST 包含 $60000$ 张官方训练图像和 $10000$ 张官方测试图像，均为 $28 times 28$ 的单通道灰度图像，类别为数字 $0$ 至 $9$。从官方训练集按类别比例划分 $54000$ 张拟合样本和 $6000$ 张验证样本，随机种子固定为 $42$。

用 $x'=x/255$ 将像素转为 $[0,1]$ 的 float32 张量，维度为 $(N,1,28,28)$；标签使用 int64。该固定变换无需估计统计量，不使用数据增强。全连接网络在前向传播时展平图像，LeNet 保留空间维度，两者接收完全相同的数据。

== 卷积网络与全连接网络

全连接网络把 $784$ 个像素作为独立输入特征，基准结构为 $784 arrow 256 arrow 10$，加深结构为 $784 arrow 256 arrow 128 arrow 10$，隐藏层均使用 ReLU，输出层产生 $10$ 维 logits。参数量分别为 $203530$ 和 $235146$。

卷积层在局部感受野内加权求和，同一组权重在不同位置共享。输出通道 $o$ 的计算可表示为

$ z_(o,i,j) = b_o + sum_(c in S_o) sum_(u,v) W_(o,c,u,v) x_(c,i+u,j+v), quad a_(o,i,j) = tanh(z_(o,i,j)). $

其中 $S_o$ 为该输出允许连接的输入通道集合。C1 只有一个输入通道，C3 使用下文的非对称连接表。LeNet 的 C1、C3 及两层隐藏全连接层使用 tanh，符合手册对 C3 非线性变换的描述；全连接对照网络保留 ReLU。

边界处按填充规则处理。特征图尺寸由

$ H_"out" = floor((H_"in" + 2P - K)/S) + 1 $

确定。S2、S4 使用$2 times 2$ 最大池化；最大池化无需可训练参数，可降低空间分辨率并增强对部分局部位移的容忍度，但不保证任意平移不变。

#figure(
    table(
        columns: (1fr, 2.1fr, 1.3fr, .9fr),
        table.header([*层*], [*设置*], [*输出维度*], [*参数量*]),
        [输入], [灰度图像], [$1 times 28 times 28$], [$0$],
        [C1 + tanh], [$1 arrow 6$，$5 times 5$，$P=2$], [$6 times 28 times 28$], [$156$],
        [S2], [最大池化，$K=S=2$], [$6 times 14 times 14$], [$0$],
        [C3 + tanh], [非对称连接，$5 times 5$，$P=0$], [$16 times 10 times 10$], [$1591$],
        [S4 / 展平], [最大池化，$K=S=2$], [$16 times 5 times 5 arrow 400$], [$0$],
        [F5 + tanh], [$400 arrow 120$], [$120$], [$48120$],
        [F6 + tanh], [$120 arrow 84$], [$84$], [$10164$],
        [F7], [$84 arrow 10$，logits], [$10$], [$850$],
        [合计], table.cell(colspan: 2)[包含所有权重与偏置], [$60881$],
    ),
    caption: [LeNet 的尺寸变化与可训练参数量],
)

全通道卷积层参数量为 $(C_"in" K^2+1)C_"out"$，全连接层为 $(n_"in"+1)n_"out"$。C3 的参数量必须按实际连接数逐输出相加，不能使用全通道卷积公式。

== C3 非对称连接的实现与核验

自定义 `C3`：通过 `ModuleList` 注册十六组独立的 `Conv2d(len(channels), 1, 5)`，每组只分配允许连接的权重和一个偏置。下表通道索引从 $0$ 开始。

#figure(
    table(
        columns: (.9fr, 2.2fr, 1fr, 1.2fr),
        table.header([*C3 输出*], [*连接的 S2 输入*], [*切片数*], [*权重及偏置*]),
        ..c3
            .connections
            .enumerate()
            .map(((output, channels)) => (
                str(output),
                channels.map(str).join(", "),
                str(channels.len()),
                str(channels.len() * 25 + 1),
            ))
            .flatten(),
        [合计], [其余 $33$ 处不连接], str(c3.kernel_slices), str(c3.parameters),
    ),
    caption: [C3 非对称连接关系],
)

特别是第 $12$、$13$、$14$ 个输出各连接 *五个* 输入通道。因此有效切片数及 C3 参数量为

$ N_"slice" = 6 times 3 + 6 times 4 + 3 times 5 + 1 times 6 = 63, $
$ N_"C3" = 63 times 5 times 5 + 16 = 1591. $

相对原全通道 C3 的 $2416$ 个参数，减少 $825$ 个，即约 #percent(825 / 2416)；整网由 $61706$ 减为 $60881$ 个参数。每个输出共享一个空间偏置，而不是给每个二维切片单独增加偏置。

前向计算时，将这 $63$ 个切片通过 `index_copy` 放到临时的 $(16,6,5,5)$ 零张量中，一次 `conv2d` 完成所有输出。该张量不是新的可训练参数；未连接位置在每次前向时均为零，优化器只更新已注册的 $1591$ 个参数，梯度通过复制操作回传到各独立卷积核。临时补零仍采用稠密卷积算子，因此参数减少不代表卷积运算量同比减少。

运行前由 `verify_c3()` 自动核验：逐输出选取输入通道卷积的参考结果与批量实现一致；输入与全部权重、偏置的梯度一致；逐一扰动 $33$ 处未连接输入时对应输出不变，且输入梯度为零；一次 Adam 更新及 `state_dict` 恢复后，未连接权重仍为零。另检查 C3 和整网参数量，以及整网输出 $(N,10)$。本次检查状态为 #if c3.passed [*通过*] else [*失败*]，结果保存在 `metrics.json`。

== 损失、训练与验证

对 logits $z$，softmax 概率、交叉熵与准确率分别为

$ p_(i,c) = exp(z_(i,c))/(sum_(k=0)^9 exp(z_(i,k))), quad L = -1/N sum_(i=1)^N ln p_(i,y_i), $
$ "Accuracy" = 1/N sum_(i=1)^N bold(1)[arg max_c z_(i,c) = y_i]. $

训练使用 `CrossEntropyLoss` 与 Adam。交叉熵直接接收 logits，网络末尾无需显式 softmax；导出预测概率时再计算 softmax。每轮打乱拟合样本，小批量执行梯度清零、前向计算、反向传播、参数更新。每轮结束后切换 `eval()`，在 `inference_mode()` 下重新评估完整拟合集和验证集，曲线不使用最后一个批次代替整轮指标。

每个配置优先选择验证准确率最高的轮次，同准确率时选择验证损失较小者，完全相同保留更早轮次。改善时立即保存权重。全部配置训练结束后，按同一规则分别选定全连接网络与 LeNet 的代表配置，然后恢复每组自己的最优权重，在官方测试集各评估一次。这里不再合并验证集重训，确保测试和可视化使用的正是验证集最优模型。

= 实验步骤：(不要求罗列完整源代码)

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.py
```

= 结论分析与体会：

== 测试性能与模型规模

#figure(
    table(
        columns: (1.65fr, 1fr, .7fr, 1fr, 1fr, 1fr),
        table.header([*配置*], [*参数量*], [*最优轮*], [*验证准确率*], [*测试损失*], [*测试准确率*]),
        ..results
            .map(result => (
                text(size: 9pt, result.name),
                str(result.parameters),
                str(result.best.epoch),
                percent(result.best.validation_accuracy),
                decimal(result.test.loss),
                percent(result.test.accuracy),
            ))
            .flatten(),
    ),
    caption: [测试结果来自各组验证集最优权重；测试集共 $10000$ 张图像],
)

验证集分别选定全连接模型 #raw(mlp.name) 与 LeNet 模型 #raw(lenet.name)。前者测试正确 #mlp.test.correct 张，准确率为 #percent(mlp.test.accuracy)；后者正确 #lenet.test.correct 张，准确率为 #percent(lenet.test.accuracy)。#if metrics.requirement_met [两者均满足超过 $97%$ 的要求。] else [本次尚未满足两类模型均超过 $97%$ 的要求。] LeNet 基准组的测试准确率为 #percent(baseline.test.accuracy)。测试结果仅用于固定方案的对照，不再反向改变验证阶段的模型选择。

LeNet 的参数量为 #lenet.parameters，是选定全连接模型的 #percent(lenet.parameters / mlp.parameters)，减少约 #percent(1 - lenet.parameters / mlp.parameters)；本次测试准确率提高 #decimal((lenet.test.accuracy - mlp.test.accuracy) * 100) 个百分点。局部连接和权重共享为手写图像提供了合适的结构约束，因此更少参数也能获得更好的分类表现。不过参数量不等于运算量，卷积权重需在多个空间位置重复使用，不能仅凭参数量断言训练一定更快。

#figure(
    image("../output/confusion_matrices.png"),
    caption: [选定全连接模型和 LeNet 的测试混淆矩阵；行是真实数字，列是预测数字],
)

#figure(
    table(
        columns: (.6fr, 1fr, 1fr, 1fr, .9fr),
        table.header([*数字*], [*精确率*], [*召回率*], [*F1*], [*测试样本数*]),
        ..range(10)
            .map(digit => {
                let row = lenet.classification_report.at(str(digit))
                (
                    str(digit),
                    decimal(row.precision),
                    decimal(row.recall),
                    decimal(row.at("f1-score")),
                    str(int(row.support)),
                )
            })
            .flatten(),
        [宏平均], decimal(lenet.classification_report.at("macro avg").precision),
        decimal(lenet.classification_report.at("macro avg").recall),
        decimal(lenet.classification_report.at("macro avg").at("f1-score")), [$10000$],
    ),
    caption: [选定 LeNet 模型的分类指标],
)

== 网络结构和超参数的影响

#figure(
    image("../output/architecture_curves.png"),
    caption: [全连接基准、加深全连接网络和 LeNet 基准的学习曲线；每个点均在该轮结束后评估],
)

#figure(
    image("../output/hyperparameter_curves.png"),
    caption: [LeNet 的轮数、学习率与批量大小对照；short 与基准的前五轮曲线重合],
)

- *训练轮数：* 五轮方案的最佳验证准确率为 #percent(short.best.validation_accuracy)，十二轮基准为 #percent(baseline.best.validation_accuracy)。短轮数的测试准确率为 #percent(short.test.accuracy)，基准为 #percent(baseline.test.accuracy)。
- *学习率：* 从 $0.001$ 降到 $0.0003$ 后，最佳验证准确率为 #percent(small.best.validation_accuracy)，测试准确率为 #percent(small.test.accuracy)。第一轮拟合损失为 #decimal(first-loss("lenet_lr_small"))，高于基准的 #decimal(first-loss("lenet"))，前期收敛更慢；在同为十二轮的预算下，最佳验证准确率也低于基准。
- *批量大小：* 从 $128$ 增至 $256$，每轮更新次数由 $ceil(54000/128)=422$ 变为 $ceil(54000/256)=211$。该配置最佳验证准确率为 #percent(large-batch.best.validation_accuracy)，测试准确率为 #percent(large-batch.test.accuracy)。批量大小变化同时改变梯度估计和总更新次数，因此相同轮数不代表相同优化步数。
- *网络深度：* 加深全连接网络使参数量由 $203530$ 增至 $235146$，测试准确率从 #percent(get-result("mlp").test.accuracy) 变为 #percent(get-result("mlp_deep").test.accuracy)。加深方案的最终拟合准确率为 #percent(get-result("mlp_deep").last.train_accuracy)，验证准确率为 #percent(get-result("mlp_deep").last.validation_accuracy)。浅层与加深方案的测试损失分别为 #decimal(get-result("mlp").test.loss) 与 #decimal(get-result("mlp_deep").test.loss)，应同时观察类别判对和概率质量。

== 首层与末层卷积可视化

使用验证集中第一个标签为 $7$ 的样本，原始训练索引为 #metrics.visualization.original_train_index（从 $0$ 开始），选定 LeNet 预测为 #metrics.visualization.predicted_label。样本选取规则在查看预测前固定。展示 tanh 之后、池化之前的特征图，C1 为 $6 times 28 times 28$，C3 为 $16 times 10 times 10$。

#figure(
    image("../output/conv1_features.png"),
    caption: [C1 的六个 $5 times 5$ 卷积核及对应激活图；左上为输入图像],
)

首层直接处理灰度像素。权重图红色表示正权重、蓝色表示负权重，同层卷积核采用相同的对称色标；tanh 特征图统一使用 $[-1,1]$ 色标，红、蓝分别表示正、负响应，接近白色表示接近零。tanh 保留负响应，因此不能套用 ReLU 的“负值被截断”解释。C1 的实际响应范围为 #decimal(metrics.visualization.conv1_activation_min) 至 #decimal(metrics.visualization.conv1_activation_max)。应结合笔画附近和背景的响应差异理解局部模式，不能仅凭一张 $5 times 5$ 权重图就把通道严格定义为某种语义检测器。

本样本中，C1 通道 $0$ 在笔画处有较完整的正响应，通道 $4$ 则在相近轮廓处呈负响应；通道 $2$、$3$ 的笔画两侧出现正负相邻的条带。这体现了不同卷积核对灰度变化及边缘方向的不同响应。

#figure(
    image("../output/conv2_kernels.png"),
    caption: [C3 的 $63$ 个有效 $5 times 5$ 切片。列为输出，行为输入，灰底 X 表示 $33$ 处未连接位置],
)

#figure(
    image("../output/conv2_features.png"),
    caption: [C3 的十六个 $10 times 10$ 激活图，通道编号与上图列编号一致],
)

C3 第 $o$ 个输出的实际可训练权重形状为 $(1, |S_o|, 5, 5)$，只对表中指定的 $3$、$4$、$5$ 或 $6$ 个输入进行卷积求和，加一个偏置后经过 tanh。图中的 $6 times 16=96$ 个格子用于呈现连接位置，其中只有 $63$ 格有权重，$33$ 格没有对应参数；第 $15$ 列连接全部六行，而第 $0$ 列只连接第 $0$、$1$、$2$ 行。导出的 `feature_maps.npz` 同时保存补零权重和布尔连接掩码，避免把不连接误解为学到了零权重。

经过 C1、S2、C3，单个 C3 激活位置对应输入上 $14 times 14$ 的理论感受野，而 C1 只有 $5 times 5$。C3 特征图更小，激活体现了指定输入通道上多个局部响应的组合，非对称连接改变通道组合而不改变该空间感受野。C3 的实际响应范围为 #decimal(metrics.visualization.conv2_activation_min) 至 #decimal(metrics.visualization.conv2_activation_max)。单样本上的弱响应不意味着该卷积核对所有数字都无用，也不能把某通道直接等同于数字 $7$ 的检测器。

例如 C3 通道 $3$、$10$ 在图像上部形成横向正响应带，通道 $12$、$15$ 则在左侧与下部形成更明显的正响应区域。与 C1 较贴近原始笔画的轮廓相比，这些低分辨率响应表现为较大区域上的组合模式。

== 实验体会

完成了 C3 连接表复现、连接隔离与梯度检查、数据完整性检查、分层划分、六组对照、验证集最优权重保存与恢复、独立测试、逐类指标及卷积可视化。此次修正说明，形状相同并不能证明网络结构相同；C3 是否正确必须同时检查连接集合、可训练参数和反向传播。选模型时还需区分准确率和交叉熵，因为二者分别侧重类别判对与概率质量。

= 就实验过程中遇到和出现的问题，你是如何解决和处理的，自拟1一3道问答题：

== 为什么第一层卷积后仍然是 $28 times 28$，全连接层输入为什么是 $400$？

C1 采用 $K=5$、$S=1$、$P=2$，输出为 $(28+4-5)+1=28$。经过第一次池化变为 $14$，C3 无填充卷积后为 $10$，再次池化为 $5$，最终 $16$ 个通道展平为 $16 times 5 times 5=400$。若遗漏第一层填充，尺寸就会改变，后续线性层会出现矩阵维度不匹配。实现中按逐层尺寸表检查，并验证前向输出为 $(N,10)$。

== 为什么有 $16$ 张 C3 特征图，图中却有 $96$ 格且只有 $63$ 格有卷积核？

图的行和列分别索引六个输入及十六个输出，用于显示全部可能的连接位置。手册只允许其中 $63$ 处连接，其余 $33$ 处用 X 标记。每个输出把自己允许的若干切片卷积结果求和并加一个偏置，因此最终仍只产生 $16$ 张特征图。卷积核是参数，特征图依赖当前输入，二者不能混为一谈。

== 为什么不能只把普通 C3 的部分权重初始化为零？

普通 `Conv2d(6,16,5)` 仍将全部 $2416$ 个参数交给优化器，初始为零的权重可能在反向传播后变为非零，连接约束随即失效。本实现仅注册允许连接的十六组卷积核，未连接位置在每次前向计算时临时补零，且通过输入扰动、梯度隔离、更新与恢复检查验证约束。
