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
#let title = "实验2：knn实现鸢尾花分类"

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

= 实验目的：

+ 理解 $K$ 近邻（K-Nearest Neighbors，KNN）算法基于距离与近邻投票进行分类的原理，掌握 $K$ 值对分类结果的影响。
+ 使用 NumPy 手动实现 KNN 分类器的 `fit` 和 `predict` 方法，完成鸢尾花三分类任务。
+ 掌握训练集、验证集和测试集的作用，按 $80%$/$20%$ 划分数据，并使用分层五折交叉验证选择 $K$ 值。
+ 理解特征标准化与数据泄漏问题，使用准确率、精确率、召回率、F1 值及混淆矩阵评估模型，并分析误分类样本。

= 实验软件和硬件环境：

- macos: m5, 26.6.2
- else: uv manage and `output/metrics.json`

= 实验原理和方法：

== 数据集与任务定义

使用 ```python sklearn.datasets.load_iris()``` 加载鸢尾花数据集。数据集共 $150$ 个样本，每个样本包含萼片长度、萼片宽度、花瓣长度、花瓣宽度四个特征，单位均为厘米。标签 $0$、$1$、$2$ 分别对应 setosa、versicolor、virginica，每类各有 $50$ 个样本。

通过分层抽样划分数据，保持各类别比例一致；固定随机种子为 $42$，使数据划分与交叉验证结果可以复现。

#figure(
    table(
        columns: (1.6fr, 1fr, 1fr, 1fr),
        table.header([*类别*], [*全部样本*], [*训练集*], [*测试集*]),
        [setosa（$0$）], $50$, $40$, $10$,
        [versicolor（$1$）], $50$, $40$, $10$,
        [virginica（$2$）], $50$, $40$, $10$,
        [合计], $150$, $120$, $30$,
    ),
    caption: [数据集的分层划分结果],
)

== 特征标准化与距离计算

KNN 依赖距离判断样本之间的相似程度。虽然四个特征的单位相同，但其数值分布和离散程度不同，因此使用 `StandardScaler` 对各维特征进行标准化：

$ z_j = (x_j - mu_j) / sigma_j $

其中，$mu_j$ 和 $sigma_j$ 分别为当前训练分区中第 $j$ 个特征的均值和标准差。交叉验证时，每一折单独在该折训练部分拟合标准化器，再转换该折验证部分；最终评估时，在完整训练集上重新拟合，并转换测试集。验证集和测试集均不参与均值、标准差的估计。

标准化后采用欧氏距离。对于两个四维向量 $bold(z)$ 和 $bold(u)$，有

$ d(bold(z), bold(u)) = sqrt(sum_(j=1)^4 (z_j - u_j)^2) $

由于平方根函数单调递增，直接计算平方距离并排序，得到的近邻次序与欧氏距离一致。

== KNN 分类规则

KNN 是基于实例的监督学习方法。`fit` 阶段保存训练特征和类别信息，不需要梯度下降等迭代优化；`predict` 阶段计算待预测样本到全部训练样本的距离，排序后取前 $K$ 个近邻，对其类别进行等权投票。

设 $N_K(bold(z))$ 为样本 $bold(z)$ 的 $K$ 个最近邻索引集合，预测结果为

$ hat(y) = arg max_(c in {0, 1, 2}) sum_(i in N_K(bold(z))) bb(1)(y_i = c) $

其中指示函数 $bb(1)$ 在条件成立时取 $1$，否则取 $0$。若多个类别票数相同，选择数值较小的类别标签；若距离相同，稳定排序保留训练样本的原始次序。$K$ 取奇数不能完全避免三分类中的平票，因此仍需明确平票处理规则。

== 五折交叉验证与参数选择

KNN 的邻居数 $K$ 与交叉验证的折数是两个不同参数。本实验固定折数为 $5$，在训练集内部比较 $K in {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}$。使用 `StratifiedKFold`，开启打乱并设置随机种子为 $42$，将 $120$ 个训练样本分为五折。每轮使用 $96$ 个样本拟合、$24$ 个样本验证，各类别分别占 $32$ 个和 $8$ 个。

对于每个候选 $K$，重复五轮训练与验证，记五折准确率为 $a_1, dots, a_5$，计算

$
    overline(a) = 1/5 sum_(r=1)^5 a_r, quad
    s = sqrt(1/5 sum_(r=1)^5 (a_r - overline(a))^2)
$

程序使用 ```python np.std(..., ddof=0)``` 计算标准差。以平均准确率最高的 $K$ 为最终参数；若均值相同，选择较小的 $K$。所有候选值共用相同的五折划分，测试集留到参数确定后进行一次最终评估。

== 性能评价指标

准确率为预测正确的样本数占总样本数的比例。对于每个类别，将其作为正类、其余类别作为负类，定义

$
    "Accuracy" = N_("correct") / N, quad
    "Precision" = ("TP") / ("TP" + "FP")
$
$
    "Recall" = ("TP") / ("TP" + "FN"), quad
    "F1" = (2 dot "Precision" dot "Recall") / ("Precision" + "Recall")
$

TP、FP、FN 分别表示真正例、假正例和假负例。宏平均对三个类别的指标取算术平均，加权平均按各类样本数加权。混淆矩阵的行表示真实类别，列表示预测类别，可用于定位具体的分类错误。

= 实验步骤：(不要求罗列完整源代码)

== 加载数据并划分训练集与测试集

加载特征和整数标签后，先划分样本索引，再根据索引取出训练、测试数据，以便在预测结果中保留原数据集的样本编号。关键代码如下：

```python
train_indices, test_indices = train_test_split(
    np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
)
```

划分后训练特征形状为 $(120, 4)$，测试特征形状为 $(30, 4)$。分层抽样保证测试集中每类均有 $10$ 个样本。

== 手动实现 KNN 分类器

`fit` 方法检查训练特征是否为非空二维数组、特征值是否有限、标签数量是否匹配、标签是否为整数，以及 $K$ 是否超过训练样本数。随后复制训练特征，通过 ```python np.unique(..., return_inverse=True)``` 保存有序类别并编码标签。

`predict` 方法检查模型是否已拟合及输入特征维数是否匹配，然后逐个处理待预测样本。核心计算如下，与实验程序中的实现一致：

```python
squared_distances = np.sum((self._x_train - sample) ** 2, axis=1)
neighbors = np.argsort(squared_distances, kind="stable")[: self.n_neighbors]
votes = np.bincount(self._encoded_y[neighbors], minlength=len(self._classes))
predictions[index] = self._classes[np.argmax(votes)]
```

其中 ```python np.bincount``` 统计各类别票数，```python np.argmax``` 返回最大票数第一次出现的位置，结合已排序的类别数组实现"平票选择较小标签"的规则。上述距离计算、近邻选择和投票均由 NumPy 完成，未调用现成的 `KNeighborsClassifier`。

== 在训练集内部选择最优 $K$

首先生成固定的五折索引；随后遍历候选 $K$，在每一折中重新创建标准化器和分类器，记录验证准确率。每折的主要操作如下：

```python
scaler = StandardScaler()
x_fit = scaler.fit_transform(x_train[fit_indices])
x_validation = scaler.transform(x_train[validation_indices])
model = KNNClassifier(n_neighbors=k).fit(x_fit, y_train[fit_indices])
predictions = model.predict(x_validation)
scores.append(float(accuracy_score(y_train[validation_indices], predictions)))
```

五折结果保存于 `output/cv_results.csv`。下表将准确率换算为百分数展示，标准差相应以百分点表示，均保留两位小数。

#figure(
    table(
        columns: (.5fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1.05fr, .9fr),
        fill: (x, y) => if y == 0 or y == 2 { luma(94%) },
        table.header([*K*], [*第1折*], [*第2折*], [*第3折*], [*第4折*], [*第5折*], [*均值*], [*标准差*]),
        $1$, $91.67$, $95.83$, $95.83$, $95.83$, $91.67$, $94.17$, $2.04$,
        $3$, $95.83$, $100.00$, $95.83$, $95.83$, $95.83$, $96.67$, $1.67$,
        $5$, $95.83$, $100.00$, $95.83$, $91.67$, $95.83$, $95.83$, $2.64$,
        $7$, $91.67$, $95.83$, $91.67$, $95.83$, $91.67$, $93.33$, $2.04$,
        $9$, $91.67$, $100.00$, $95.83$, $95.83$, $95.83$, $95.83$, $2.64$,
        $11$, $95.83$, $100.00$, $87.50$, $95.83$, $95.83$, $95.00$, $4.08$,
        $13$, $91.67$, $100.00$, $91.67$, $95.83$, $95.83$, $95.00$, $3.12$,
        $15$, $91.67$, $91.67$, $91.67$, $91.67$, $95.83$, $92.50$, $1.67$,
        $17$, $95.83$, $91.67$, $91.67$, $87.50$, $95.83$, $92.50$, $3.12$,
        $19$, $95.83$, $87.50$, $87.50$, $87.50$, $100.00$, $91.67$, $5.27$,
    ),
    caption: [不同 $K$ 值的分层五折验证结果],
)

#figure(
    image("../output/cv_accuracy.png"),
    caption: [交叉验证准确率随 $K$ 的变化，误差棒为五折准确率的正负一个标准差],
)

$K=3$ 的平均验证准确率最高，为 $96.67%$，标准差为 $1.67$ 个百分点，因此选择 $K=3$。其五折分别正确分类 $23$、$24$、$23$、$23$、$23$ 个验证样本，合计为 $116/120$。

== 拟合最终模型并保存测试结果

确定 $K$ 后，在完整的 $120$ 个训练样本上重新拟合标准化器与分类器，再预测 $30$ 个测试样本：

```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
final_model = KNNClassifier(n_neighbors=best.k).fit(x_train_scaled, y_train)
predictions = final_model.predict(x_test_scaled)
```

程序计算测试准确率、分类报告和混淆矩阵，并将结果保存到 `output` 目录。其中，`metrics.json` 记录配置、环境、划分索引及各项指标；`test_predictions.csv` 记录原始样本索引、四个特征、真实标签、预测标签及是否正确；`classification_report.txt` 保存分类指标；两张 PNG 分别展示交叉验证曲线和测试集混淆矩阵。

= 结论分析与体会：

== 测试集分类性能

最终模型使用 $K=3$，在测试集上正确分类 $28$ 个样本、误分类 $2$ 个样本，准确率为 $28/30 approx 93.33%$。分类报告如下，指标保留四位小数：

#figure(
    table(
        columns: (1.5fr, 1fr, 1fr, 1fr, .8fr),
        table.header([*类别／平均方式*], [*精确率*], [*召回率*], [*F1*], [*样本数*]),
        [setosa], $1.0000$, $1.0000$, $1.0000$, $10$,
        [versicolor], $0.8333$, $1.0000$, $0.9091$, $10$,
        [virginica], $1.0000$, $0.8000$, $0.8889$, $10$,
        [宏平均], $0.9444$, $0.9333$, $0.9327$, $30$,
        [加权平均], $0.9444$, $0.9333$, $0.9327$, $30$,
    ),
    caption: [$K=3$ 时的测试集分类报告],
)

#figure(
    image("../output/confusion_matrix.png"),
    caption: [测试集混淆矩阵（行：真实类别；列：预测类别）],
)

混淆矩阵按 setosa、versicolor、virginica 的顺序排列，为
$ mat(10, 0, 0; 0, 10, 0; 0, 2, 8). $
setosa 和 versicolor 的真实样本各 $10$ 个，全部识别正确；virginica 中 $8$ 个识别正确，另 $2$ 个被判为 versicolor。因此，versicolor 的召回率为 $1$，但预测为该类的 $12$ 个样本中只有 $10$ 个正确，精确率为 $10/12 approx 0.8333$；virginica 的精确率为 $1$，召回率为 $8/10 = 0.8$。三个类别的测试样本数相同，因此宏平均与加权平均一致。

== 误分类样本分析

根据 `test_predictions.csv`，错误发生在原数据集索引为 $138$ 和 $134$ 的两个样本上。索引从 $0$ 开始，表中特征均为标准化前的原始值，单位为厘米。

#figure(
    table(
        columns: (.6fr, .8fr, .8fr, .8fr, .8fr, 1.2fr, 1.2fr),
        table.header([*索引*], [*萼片长*], [*萼片宽*], [*花瓣长*], [*花瓣宽*], [*真实类别*], [*预测类别*]),
        $138$, $6.0$, $3.0$, $4.8$, $1.8$, [virginica], [versicolor],
        $134$, $6.1$, $2.6$, $5.6$, $1.4$, [virginica], [versicolor],
    ),
    caption: [测试集中的两个误分类样本],
)

为解释这两次预测，按 `metrics.json` 中保存的训练索引重新计算最终标准化空间内的近邻，得到以下结果。距离为欧氏距离，近邻按距离由小到大排列：

- 样本 $138$：近邻为 $149$（virginica，$0.2080$）、$70$（versicolor，$0.4634$）、$78$（versicolor，$0.4849$）。最近的一个样本属于 virginica，但另外两个属于 versicolor，等权投票最终以 $2$∶$1$ 判为 versicolor。
- 样本 $134$：近邻为 $83$（versicolor，$0.4630$）、$72$（versicolor，$0.5315$）、$133$（virginica，$0.5962$），同样以 $2$∶$1$ 判为 versicolor。

这说明在本次训练划分和标准化距离下，两例样本的局部邻域均包含不同类别，而等权多数投票使 versicolor 占优。单个最近邻与最终投票结果可能不同，$K$ 的选择会直接改变局部决策。不能仅依据这两个测试样本重新选择 $K$，否则会使测试集参与调参。

== K 值的影响与结果的局限

本次 $K=1$ 的平均验证准确率为 $94.17%$，$K=3$ 提高到 $96.67%$；$K=5$ 和 $K=9$ 均为 $95.83%$，$K=19$ 降至 $91.67%$。验证准确率并不随 $K$ 单调变化。在一般情况下，较小的 $K$ 更依赖局部样本、容易受到噪声影响；较大的 $K$ 会扩大投票邻域，可能混入其他类别并使决策边界过于平滑。本次结果支持在给定候选范围和划分下选择 $K=3$，但不能据此认为 $K=3$ 对所有数据划分都最优。

测试准确率比最优交叉验证均值低约 $3.33$ 个百分点。交叉验证使用的训练样本数、验证样本和最终测试样本不同，且最高交叉验证均值参与了参数选择，因此二者不必相同。测试集仅有 $30$ 个样本，每错一个样本，准确率便变化约 $3.33$ 个百分点；本次结果反映了固定划分下的表现，不能仅凭这一差值断言模型明显过拟合。

图中的误差棒表示五折分数的离散程度，并不是泛化准确率的置信区间。若要进一步比较方法，可在训练集内尝试更多 $K$ 值、距离加权投票或其他距离度量，并采用重复分层交叉验证考察稳定性；这些属于后续改进方向，本次结果中未包含相应对比实验。

== 实验体会

通过手动实现 KNN，可以清楚地对应"计算距离—选择近邻—统计票数—输出类别"的完整过程。KNN 的拟合阶段主要保存数据，计算开销集中在预测阶段。本实现对每个待测样本遍历全部训练样本并完整排序，若训练样本数为 $n$、特征数为 $d$，单个样本的预测时间复杂度约为 $O(n d + n log n)$，适合本实验这样的小规模数据集；更大规模任务可考虑部分选择或近邻检索结构。

实验还表明，可靠的评价需要完整的数据处理流程：分层抽样保持类别比例，折内标准化避免数据泄漏，交叉验证承担调参任务，独立测试集检验最终模型。准确率给出总体表现，分类报告与混淆矩阵则揭示了 virginica 召回率偏低这一具体问题。保留随机种子、数据索引和逐样本预测，也使结果能够被核对和解释。

= 就实验过程中遇到和出现的问题，你是如何解决和处理的，自拟1一3道问答题：

== 为什么不能先对全部数据标准化，再划分训练集和测试集？

答：标准化需要估计各特征的均值和标准差。若先处理全部数据，测试集的信息就会参与训练预处理，造成数据泄漏；若在完整训练集上标准化后再交叉验证，也会把各折验证部分的信息带入拟合过程。处理方法是先划分数据，并在每一折内部仅对该折训练部分调用 `fit_transform`，对验证部分调用 `transform`。确定 $K$ 后，再在完整训练集上拟合新的标准化器，并用其转换测试集。

== K 取奇数后还会出现平票吗？程序怎样处理？

答：会。奇数 $K$ 可以避免二分类中的等票，但本实验是三分类，例如 $K=3$ 时可能出现三个类别各 $1$ 票。程序先用 ```python np.unique``` 得到按标签递增排列的类别，再通过 ```python np.bincount``` 计票，使用 ```python np.argmax``` 选择最大票数的首个位置，从而在平票时选择较小标签。对于距离相同的样本，使用稳定排序保持次序，使结果确定且可复现。该规则解决了预测不唯一的问题，但带有对较小标签的偏好。

== 为什么用交叉验证选择 $K$，而不直接选择测试准确率最高的 $K$？

答：验证数据用于模型选择，测试数据用于评价选择完成后的模型。如果不断比较测试准确率并据此改变 $K$，测试集就实际承担了验证集的角色，最终分数可能过于乐观。本实验将测试集固定保留，仅在 $120$ 个训练样本内部完成五折验证，按平均准确率选择 $K=3$，再用完整训练集拟合最终模型并测试一次。测试中的两例错误用于分析模型局限，不作为重新调参的依据。
