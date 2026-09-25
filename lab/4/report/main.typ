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
#let percent(value) = eval(mode: "math", str(calc.round(value * 100, digits: 2)) + "%")
#let decimal(value) = eval(mode: "math", str(calc.round(value, digits: 4)))

= 实验目的：

+ 理解卷积、参数共享、池化与非线性激活的作用，实现 LeNet 手写数字分类。
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

$ z_(o,i,j) = b_o + sum_c sum_(u,v) W_(o,c,u,v) x_(c,i+u,j+v), quad a_(o,i,j) = max(0, z_(o,i,j)). $

边界处按填充规则处理。特征图尺寸由

$ H_"out" = floor((H_"in" + 2P - K)/S) + 1 $

确定。使用 ReLU 和 $2 times 2$ 最大池化；最大池化无需可训练参数，可降低空间分辨率并增强对部分局部位移的容忍度，但不保证任意平移不变。

#figure(
    table(
        columns: (1fr, 2.1fr, 1.3fr, .9fr),
        table.header([*层*], [*设置*], [*输出维度*], [*参数量*]),
        [输入], [灰度图像], [$1 times 28 times 28$], [$0$],
        [C1 + ReLU], [$1 arrow 6$，$5 times 5$，$P=2$], [$6 times 28 times 28$], [$156$],
        [S2], [最大池化，$K=S=2$], [$6 times 14 times 14$], [$0$],
        [C3 + ReLU], [$6 arrow 16$，$5 times 5$，$P=0$], [$16 times 10 times 10$], [$2416$],
        [S4 / 展平], [最大池化，$K=S=2$], [$16 times 5 times 5 arrow 400$], [$0$],
        [F5 + ReLU], [$400 arrow 120$], [$120$], [$48120$],
        [F6 + ReLU], [$120 arrow 84$], [$84$], [$10164$],
        [F7], [$84 arrow 10$，logits], [$10$], [$850$],
        [合计], table.cell(colspan: 2)[包含所有权重与偏置], [$61706$],
    ),
    caption: [LeNet 的尺寸变化与可训练参数量],
)

卷积层参数量为 $(C_"in" K^2+1)C_"out"$，全连接层为 $(n_"in"+1)n_"out"$。例如 C3 的参数量为 $(6 times 5 times 5+1)times 16=2416$。使用 PyTorch 的全通道连接而非原始 LeNet 不对称连接；同时使用最大池化代替图示的平均池化。因此这是用于 MNIST 的 LeNet 变体，不能将其参数量解释为原始 LeNet-5 的参数量。

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

验证集分别选定全连接模型 #raw(mlp.name) 与 LeNet 模型 #raw(lenet.name)。前者测试正确 #mlp.test.correct 张，准确率为 #percent(mlp.test.accuracy)；后者正确 #lenet.test.correct 张，准确率为 #percent(lenet.test.accuracy)。两者均满足超过 $97%$ 的要求。LeNet 基准组的测试准确率为 #percent(baseline.test.accuracy)，高于验证选中的批量 $256$ 配置；但测试排序不用于反向改变选择，这也说明验证集与测试集上的排序可能不同。

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
- *学习率：* 从 $0.001$ 降到 $0.0003$ 后，最佳验证准确率为 #percent(small.best.validation_accuracy)，测试准确率为 #percent(small.test.accuracy)。第一轮拟合损失为 $0.2715$，高于基准的 $0.1358$，表明前期收敛更慢；十二轮结束时最佳验证准确率也较低。
- *批量大小：* 从 $128$ 增至 $256$，每轮更新次数由 $ceil(54000/128)=422$ 变为 $ceil(54000/256)=211$。该配置最佳验证准确率为 #percent(large-batch.best.validation_accuracy)，测试准确率为 #percent(large-batch.test.accuracy)。批量大小变化同时改变梯度估计和总更新次数，因此相同轮数不代表相同优化步数。
- *网络深度：* 加深全连接网络使参数量由 $203530$ 增至 $235146$，测试准确率从 #percent(get-result("mlp").test.accuracy) 变为 #percent(get-result("mlp_deep").test.accuracy)。加深方案的最终拟合准确率为 $99.86%$，验证准确率为 $97.87%$，仍有泛化差距；其测试损失也高于浅层方案，说明准确率提高不意味着概率预测一定更好。

== 首层与末层卷积可视化

使用验证集中第一个标签为 $7$ 的样本，原始训练索引为 #metrics.visualization.original_train_index（从 $0$ 开始），选定 LeNet 预测为 #metrics.visualization.predicted_label。样本选取规则在查看预测前固定。展示 ReLU 之后、池化之前的特征图，C1 为 $6 times 28 times 28$，C3 为 $16 times 10 times 10$。

#figure(
    image("../output/conv1_features.png"),
    caption: [C1 的六个 $5 times 5$ 卷积核及对应激活图；左上为输入图像],
)

首层直接处理灰度像素。权重图红色表示正权重、蓝色表示负权重，同层所有卷积核采用相同的对称色标；特征图用相同的 $[0, max]$ 色标，亮色表示较强响应。通道 $0$、$1$、$5$ 保留较完整的笔画轮廓，通道 $2$、$3$ 对局部边缘响应更集中，说明卷积学习到了互补的局部特征。ReLU 将负响应截断为零，背景或不匹配位置的响应较弱。不能仅凭一张 $5 times 5$ 权重图就把某个通道严格定义为某种语义检测器。

#figure(
    image("../output/conv2_kernels.png"),
    caption: [末层 C3 的全部卷积核切片。列对应 $16$ 个输出通道，行对应 $6$ 个输入通道；共 $96$ 个 $5 times 5$ 切片],
)

#figure(
    image("../output/conv2_features.png"),
    caption: [C3 的十六个 $10 times 10$ 激活图，通道编号与上图列编号一致],
)

C3 的每个输出卷积核实际是 $6 times 5 times 5$ 的三维权重，输出特征图由六个输入通道卷积结果求和、加偏置、再经过 ReLU 得到。上图完整展示各输入通道切片，避免只展示均值而掩盖通道间差异。

经过 C1、S2、C3，单个 C3 激活位置对应输入上 $14 times 14$ 的理论感受野，而 C1 只有 $5 times 5$。C3 特征图更小，激活体现了多个局部响应的组合；通道 $7$、$9$ 的较强响应主要集中于上方横向区域，通道 $10$ 则对两侧局部区域响应明显。单样本上的弱响应不意味着该卷积核对所有数字都无用，也不能把某通道直接等同于数字 $7$ 的检测器。

== 实验体会与局限

完成了数据完整性检查、分层划分、六组对照、验证集最优权重保存与恢复、独立测试、逐类指标及卷积可视化。CNN 利用图像的空间结构，在本次 MNIST 中以更少参数获得了较好的测试表现；选模型时还需区分准确率和交叉熵，因为二者分别侧重类别判对与概率质量。

仅使用一个固定随机种子和一次数据划分，配置间较小差异可能受随机初始化和训练波动影响，不能据此断言某超参数普遍最优。MNIST 图像较简单，结果也不能直接外推到复杂自然图像；特征可视化反映局部响应而非完整的因果解释。

= 就实验过程中遇到和出现的问题，你是如何解决和处理的，自拟1一3道问答题：

== 为什么第一层卷积后仍然是 $28 times 28$，全连接层输入为什么是 $400$？

C1 采用 $K=5$、$S=1$、$P=2$，输出为 $(28+4-5)+1=28$。经过第一次池化变为 $14$，C3 无填充卷积后为 $10$，再次池化为 $5$，最终 $16$ 个通道展平为 $16 times 5 times 5=400$。若遗漏第一层填充，尺寸就会改变，后续线性层会出现矩阵维度不匹配。实现中按逐层尺寸表检查，并验证前向输出为 $(N,10)$。

== 为什么末层有 $16$ 张特征图，却展示了 $96$ 张卷积核切片？

末层有 $16$ 个输出通道，每个输出卷积核连接全部 $6$ 个输入通道，因此权重张量是 $(16,6,5,5)$。可视化时拆成 $16 times 6=96$ 个二维切片，但计算时每组 $6$ 个切片的卷积结果先求和，最终只产生 $16$ 张特征图。卷积核是训练得到的参数，特征图则依赖当前输入图像，两者不能混为一谈。
