# 数据科学的通用框架

目录结构：

![image_1](./pic/image_1.png)

强调下面几个独立的仓库，可供学习参考。

| 名称                                      | 地址                                                         | 说明                                                         |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Awesome Machine Learning Interpretability | https://github.com/jphall663/awesome-machine-learning-interpretability | 很棒的机器学习可解释性资源列表。                             |
| Awesome Business Machine Learning         | https://github.com/firmai/business-machine-learning          | 应用商业机器学习 (BML) 和商业数据科学 (BDS) 示例和库的精选列表。 |
| Awesome Industry Machine Learning         | https://github.com/firmai/industry-machine-learning          | 不同行业的应用机器学习和数据科学和库的精选列表。（notebook） |
| Awesome Learning with Label Noise         | https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise | 使用噪声标签学习的精选资源列表。                             |
| Awesome Time Series                       | https://github.com/MaxBenChrist/awesome_time_series_in_python | 用于处理时间序列的各种 Python 包。                           |
| Awesome Time Series Anomaly Detection     | https://github.com/rob-med/awesome-TS-anomaly-detection      | 用于时间序列数据异常检测的工具和数据集列表。                 |
| Awesome Causality                         | https://github.com/rguo12/awesome-causality-algorithms       | 用数据学习因果关系的算法列表。                               |





### 模型搭建

#### 可解释性模型

| 名称                                | 地址                                                        | 说明                                                         |
| ----------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| acd                                 | https://github.com/csinva/hierarchical_dnn_interpretations  | 为 pytorch 神经网络做出的单个预测生成分层解释                |
| aequitas                            | https://github.com/dssg/aequitas                            | Aequitas 是一个开源偏见审计工具包，供数据科学家、机器学习研究人员和政策制定者审计机器学习模型的歧视和偏见，并围绕开发和部署预测工具做出明智和公平的决策 |
| AI Fairness 360                     | https://github.com/Trusted-AI/AIF360                        | AI Fairness 360 工具包是一个可扩展的开源库，其中包含由研究社区开发的技术，可帮助在整个 AI 应用程序生命周期中检测和减轻机器学习模型中的偏差 |
| AI Explainability 360               | https://github.com/IBM/AIX360                               | AI Explainability 360 工具包是一个开源库，支持数据集和机器学习模型的可解释性和可解释性。 AI Explainability 360 包括一套全面的算法，涵盖不同维度的解释以及代理可解释性指标 |
| ALEPython                           | https://github.com/blent-ai/ALEPython                       | 计算并绘制累积局部效果（ALE）                                |
| Aletheia                            | https://github.com/SelfExplainML/Aletheia                   | 用于展开 ReLU 神经网络                                       |
| allennlp                            | https://github.com/allenai/allennlp                         | 基于 PyTorch 构建的 Apache 2.0 NLP 研究库，用于在各种语言任务上开发最先进的深度学习模型 |
| algofairness                        | https://github.com/algofairness                             |                                                              |
| Alibi                               | https://github.com/SeldonIO/alibi                           | Alibi 是一个针对机器学习模型检查和解释的开源 Python 库。 该库的重点是为分类和回归模型提供黑盒、白盒、局部和全局解释方法的高质量实现 |
| anchor                              | https://github.com/marcotcr/anchor                          | 使用锚点来解释文本分类器或作用于数据表的分类器的单个预测     |
| BlackBoxAuditing                    | https://github.com/algofairness/BlackBoxAuditing            | 梯度特征审计 (GFA) 的示例实现                                |
| casme                               | https://github.com/kondiz/casme                             | 在 ImageNet 上使用与分类器无关的显著图提取的示例             |
| Causal Discovery Toolbox            | https://github.com/FenTechSolutions/CausalDiscoveryToolbox  | Causal Discovery Toolbox 是一个用于在图形和 Python>=3.5 的成对设置中进行因果推理的包， 包括用于图结构恢复和依赖关系的工具。 |
| captum                              | https://github.com/pytorch/captum                           | Captum 是 PyTorch 的模型可解释性和理解库，包含用于 PyTorch 模型的集成梯度、显着性图、smoothgrad、vargrad 等的通用实现。 它可以快速集成使用特定领域库（如 torchvision、torchtext 等）构建的模型。 |
| causalml                            | https://github.com/uber/causalml                            | 一个 Python 包，它提供了一套使用基于最近研究的机器学习算法的提升建模和因果推理方法。 它提供了一个标准界面，允许用户根据实验或观察数据估计条件平均治疗效果 (CATE) 或个体治疗效果 (ITE)。 |
| cdt15                               | https://github.com/cdt15                                    |                                                              |
| checklist                           | https://github.com/marcotcr/checklist                       | 该存储库包含用于测试 NLP 模型的代码                          |
| contextual-AI                       | https://github.com/SAP/contextual-ai                        | 上下文 AI 为机器学习管道的不同阶段（数据、训练和推理）增加了可解释性，从而解决了此类 ML 系统与其用户之间的信任差距 |
| ContrastiveExplanation (Foil Trees) | https://github.com/MarcelRobeer/ContrastiveExplanation      | 对比解释解释了为什么一个实例具有当前结果（fact）而不是感兴趣的目标结果（foil）。 这些反事实解释将解释限制在与区分 fact 和 foil 相关的特征上，从而忽略了不相关的特征 |
| counterfit                          | https://github.com/Azure/counterfit/                        | 用于评估机器学习系统安全性的命令行工具和通用自动化层         |
| dalex                               | https://github.com/ModelOriented/DALEX                      | 用于探索和解释的模型无关语言，对任何模型进行 X 射线扫描，有助于探索和解释其行为，有助于了解复杂模型的工作原理 |
| debiaswe                            | https://github.com/tolga-b/debiaswe                         | 从词嵌入中消除有问题的性别偏见                               |
| DeepExplain                         | https://github.com/marcoancona/DeepExplain                  | 最先进的梯度和基于扰动的归因方法的统一框架。 研究人员和从业者可以使用它来更好地理解推荐系统的现有模型，以及对其他归因方法进行基准测试 |
| deeplift                            | https://github.com/kundajelab/deeplift                      | 深度学习的重要特征                                           |
| deepvis                             | https://github.com/yosinski/deep-visualization-toolbox      | 通过深度可视化理解神经网络                                   |
| DiCE                                | https://github.com/interpretml/DiCE                         | 为任何机器学习模型生成不同的反事实解释                       |
| DoWhy                               | https://github.com/microsoft/dowhy                          | 因果推理的统一语言，结合了因果图形模型和潜在结果框架，支持显式建模和因果假设测试 |
| ecco                                | https://github.com/jalammar/ecco                            | 使用交互式可视化探索和解释自然语言处理模型的库               |
| eli5                                | https://github.com/TeamHG-Memex/eli5                        | 有助于调试机器学习分类器并解释它们的预测                     |
| explainerdashboard                  | https://github.com/oegedijk/explainerdashboard              | 快速构建可解释的 AI 仪表板，显示所谓的“黑盒”机器学习模型的内部工作原理 |
| fairml                              | https://github.com/adebayoj/fairml                          | 审核机器学习模型偏差的工具箱                                 |
| fairlearn                           | https://github.com/fairlearn/fairlearn                      | 该软件包使人工智能 (AI) 系统的开发人员能够评估其系统的公平性并减轻任何观察到的不公平问题，同时包含缓解算法以及模型评估的指标 |
| fairness-comparison                 | https://github.com/algofairness/fairness-comparison         | 该存储库旨在促进公平感知机器学习算法的基准测试               |
| fairness_measures_code              | https://github.com/megantosh/fairness_measures_code         | 量化歧视的措施实现                                           |
| foolbox                             | https://github.com/bethgelab/foolbox                        | Foolbox 是一个 Python 库，可让您轻松地对深度神经网络等机器学习模型进行对抗性攻击。 它建立在 EagerPy 之上，可与 PyTorch、TensorFlow 和 JAX 中的模型原生配合使用 |
| Grad-CAM                            | https://github.com/topics/grad-cam                          |                                                              |
| gplearn                             | https://github.com/trevorstephens/gplearn                   | Python 中的遗传编程，并带有受 scikit-learn 启发的 API        |
| hate-functional-tests               | https://github.com/paul-rottger/hate-functional-tests       | 仇恨言论检测模型的功能测试                                   |
| human-learn                         | https://github.com/koaning/human-learn                      |                                                              |
| imodels                             | https://github.com/csinva/imodels                           | 可解释的 ML 包，用于简洁、透明和准确的预测建模（与 sklearn 兼容） |
| iNNvestigate neural nets            | https://github.com/albermax/innvestigate                    | 目标是让分析神经网络的预测变得容易                           |
| Integrated-Gradients                | https://github.com/ankurtaly/Integrated-Gradients           | 研究将深度网络的预测归因于其输入特征的问题，作为解释个体预测的尝试 |
| interpret                           | https://github.com/interpretml/interpret                    | 整合了最先进的机器学习可解释性技术，可以训练可解释的白盒模型并解释黑盒系统，帮助了解模型的全局行为，或了解个别预测背后的原因 |
| interpret_with_rules                | https://github.com/clips/interpret_with_rules               | 引入规则来解释经过训练的神经网络的预测，并可选地解释模型从训练数据中捕获的模式以及原始数据集中存在的模式 |
| Keras-vis                           | https://github.com/raghakot/keras-vis                       | 用于可视化和调试经过训练的 keras 神经网络模型                |
| keract                              | https://github.com/philipperemy/keract/                     |                                                              |
| L2X                                 | https://github.com/Jianbo-Lab/L2X                           | 学习解释：模型解释的信息论视角                               |
| lime                                | https://github.com/marcotcr/lime                            | 解释机器学习分类器（或模型）在做什么，目前支持解释文本分类器、表格或图像的分类器的单个预测 |
| LiFT                                | https://github.com/linkedin/LiFT                            | LinkedIn Fairness Toolkit (LiFT) 是一个 Scala/Spark 库，可用于测量大规模机器学习工作流中的公平性 |
| LORE                                | https://github.com/riccotti/LORE                            | 此存储库包含 LORE（基于本地规则的解释）的源代码              |
| lit                                 | https://github.com/pair-code/lit                            | 语言可解释性工具：交互式分析 NLP 模型以在可扩展且与框架无关的界面中理解模型 |
| lofo-importance                     | https://github.com/aerdem4/lofo-importance                  | 通过迭代地从集合中删除每个特征，并使用验证方案评估模型的性能，根据选择的度量计算一组特征的重要性 |
| lrp_toolbox                         | https://github.com/sebastian-lapuschkin/lrp_toolbox         | LRP 工具箱为支持 Matlab 和 Python 的人工神经网络提供简单且可访问的 LRP 独立实现。 |
| MindsDB                             | https://github.com/mindsdb/mindsdb                          | MindsDB 使您能够使用 SQL 在数据库中使用 ML 预测              |
| MLextend                            | http://rasbt.github.io/mlxtend/                             | 作为 sklearn的一个扩展包，可以快速组装模型融合，并集成了从数据到特征选择、建模、验证和可视化的一套完整的工作流 |
| ml-fairness-gym                     | https://github.com/google/ml-fairness-gym                   | 一组用于构建简单模拟的组件，用于探索在社会环境中部署基于机器学习的决策系统的潜在长期影响 |
| ml_privacy_meter                    | https://github.com/privacytrustlab/ml_privacy_meter         | 一种量化机器学习模型在推理攻击方面的隐私风险的工具，特别是成员推理攻击 |
| OptBinning                          | https://github.com/guillermo-navas-palencia/optbinning      | OptBinning 是一个用 Python 编写的库，实现了一个严格而灵活的数学编程公式，以解决二进制、连续和多类目标类型的最佳分箱问题，并结合了以前未解决的约束 |
| parity-fairness                     | https://pypi.org/project/parity-fairness/                   | 该存储库包含演示使用公平指标、偏见缓解和可解释性工具的代码   |
| PDPbox                              | https://github.com/SauceCat/PDPbox                          | 部分依赖图工具箱（PDP）                                      |
| pyBreakDown                         | https://github.com/MI2DataLab/pyBreakDown                   | 一种与模型无关的工具，用于从黑盒中分解预测。 分解表显示了每个变量对最终预测的贡献，并以简洁的图形方式呈现变量贡献。适用于二元分类器和一般回归模型 |
| PyCEbox                             | https://github.com/AustinRochford/PyCEbox                   | 个体条件期望图的实现                                         |
| pyGAM                               | https://github.com/dswah/pyGAM                              | 广义加法模型                                                 |
| pymc3                               | https://github.com/pymc-devs/pymc3                          | PyMC（以前称为 PyMC3）是一个用于贝叶斯统计建模的 Python 包，专注于高级马尔可夫链蒙特卡罗（MCMC）和变分推理（VI）算法。 它的灵活性和可扩展性使其适用于大量问题。 |
| pytorch-innvestigate                | https://github.com/fgxaos/pytorch-innvestigate              |                                                              |
| pyss3                               | https://github.com/sergioburdisso/pyss3                     | 一个 Python 包，实现了一种用于文本分类的新的可解释机器学习模型 |
| rationale                           | https://github.com/taolei87/rcnn/tree/master/code/rationale | 该方法学习提供理由，即基本原理，作为神经网络预测的支持证据   |
| responsibly                         | https://github.com/ResponsiblyAI/responsibly                | 目标是一站式审计机器学习系统的偏见和公平性，次要目标是通过算法干预减轻偏见和调整公平性。 此外，还特别关注 NLP 模型。 |
| revise-tool                         | https://github.com/princetonvisualai/revise-tool            | 一种工具，可沿基于对象、基于属性和基于地理的模式的轴自动检测视觉数据集中可能存在的偏差形式，并从中提出缓解措施的建议。 |
| robustness                          | https://github.com/MadryLab/robustness                      | 用于使训练、评估和探索神经网络变得灵活和容易                 |
| risk-slim                           | https://github.com/ustunb/risk-slim                         | 一种机器学习方法，用于在 python 中拟合简单的自定义风险评分。 |
| RISE                                | https://github.com/eclique/RISE                             | 检测模型的注意力                                             |
| sage                                | https://github.com/iancovert/sage/                          | SAGE（Shapley Additive GlobalimportancE）是一种博弈论方法，用于理解黑盒机器学习模型。 它根据每个特征贡献的预测能力总结每个特征的重要性，并使用 Shapley 值解释复杂的特征交互 |
| SALib                               | https://github.com/SALib/SALib                              | 常用敏感性分析方法的 Python 实现， 在系统建模中用于计算模型输入或外生因素对感兴趣输出的影响 |
| scikit-fairness                     | https://github.com/koaning/scikit-fairness                  |                                                              |
| sklearn-expertsys                   | https://github.com/tmadl/sklearn-expertsys                  | scikit learn 的高度可解释分类器，生成易于理解的决策规则，而不是黑盒模型 |
| shap                                | https://github.com/slundberg/shap                           | SHAP（SHapley Additive exPlanations）是一种博弈论方法，用于解释任何机器学习模型的输出， 它使用博弈论中的经典 Shapley 值及其相关扩展将最优信用分配与局部解释联系起来 |
| shapley                             | https://github.com/benedekrozemberczki/shapley              | 该库包含各种计算（近似）加权投票游戏（集成游戏）中玩家（模型）的 Shapley 值的方法 - 一类可转移的效用合作游戏 |
| Skater                              | https://github.com/datascienceinc/Skater                    | Skater 是一个统一的框架，可以对所有形式的模型进行模型解释，以帮助构建现实世界用例经常需要的可解释机器学习系统，旨在揭开全局（基于完整数据集的推理）和局部（关于单个预测的推理）的黑盒模型的学习结构的神秘面纱 |
| tensorfow/cleverhans                | https://github.com/tensorflow/cleverhans                    | 该库专注于提供针对机器学习模型的攻击的参考实现，以帮助针对对抗性示例对模型进行基准测试 |
| tensorflow/lucid                    | https://github.com/tensorflow/lucid                         | 用于研究神经网络可解释性的基础设施和工具的集合               |
| tensorflow/fairness-indicators      | https://github.com/tensorflow/fairness-indicators           | Fairness Indicators 旨在支持团队与更广泛的 Tensorflow 工具包合作评估、改进和比较公平问题模型 |
| tensorflow/model-analysis           | https://github.com/tensorflow/model-analysis                | TensorFlow 模型分析 (TFMA) 是一个用于评估 TensorFlow 模型的库。 它允许用户使用在他们的培训师中定义的相同指标，以分布式方式评估他们对大量数据的模型。 这些指标可以在不同的数据切片上计算，并在 Jupyter 笔记本中可视化 |
| tensorflow/model-card-toolkit       | https://github.com/tensorflow/model-card-toolkit            | 用于自动生成模型卡，这是一种机器学习文档，可为模型的开发和性能提供上下文和透明度 |
| tensorflow/model-remediation        | https://github.com/tensorflow/model-remediation             | TensorFlow Model Remediation 是一个库，它为机器学习从业者提供解决方案，以减少或消除潜在性能偏差导致的用户伤害，从而创建和训练模型。 |
| tensorflow/privacy                  | https://github.com/tensorflow/privacy                       | 该存储库包含 TensorFlow Privacy 的源代码，这是一个 Python 库，其中包含 TensorFlow 优化器的实现，用于训练具有差异隐私的机器学习模型。 该库附带用于计算提供的隐私保证的教程和分析工具。 |
| tensorflow/tcav                     | https://github.com/tensorflow/tcav                          | 超越特征归因的可解释性：使用概念激活向量 (TCAV) 进行定量测试 |
| tensorfuzz                          | https://github.com/brain-research/tensorfuzz                | 一个用于执行神经网络覆盖引导模糊测试的库。                   |
| TensorWatch                         | https://github.com/microsoft/tensorwatch                    | TensorWatch 是微软研究院为数据科学、深度学习和强化学习设计的调试和可视化工具，它在 Jupyter Notebook 中工作。 |
| TextFooler                          | https://github.com/jind11/TextFooler                        | 文本分类和推理的自然语言攻击模型                             |
| tf-explain                          | https://github.com/sicara/tf-explain                        | tf-explain 将可解释性方法实现为 Tensorflow 2.x 回调，以简化神经网络的理解 |
| Themis                              | https://github.com/LASER-UMASS/Themis                       | 一种基于测试的方法，用于测量软件系统中的歧视                 |
| themis-ml                           | https://github.com/cosmicBboy/themis-ml                     | 建立在 pandas 和 sklearn 之上的 Python 库，它实现了公平感知机器学习算法。 |
| treeinterpreter                     | https://github.com/andosa/treeinterpreter                   | 用于解释 scikit-learn 的决策树和随机森林预测的库             |
| woe                                 | https://github.com/boredbird/woe                            | WOE 主要用于信用评级记分卡模型                               |
| xai                                 | https://github.com/EthicalML/xai                            | XAI 是一个机器学习库，其设计以 AI 可解释性为核心。 XAI 包含各种工具，可用于分析和评估数据和模型。 |
| xdeep                               | https://github.com/datamllab/xdeep                          | XDeep 是一个用于可解释机器学习的开源 Python 库。 它由德克萨斯 A&M 大学的 DATA 实验室开发。 XDeep 的目标是为想要了解深度模型如何工作的人们提供易于访问的解释工具。 XDeep 提供了多种方法来解释本地和全局模型。 |
| yellowbrick                         | https://github.com/DistrictDataLabs/yellowbrick             | 便于机器学习模型选择的可视化分析和诊断工具。                 |

#### 概率统计模型

| 名称                   | 地址                                                         | 说明                                 |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------ |
| PyMC3                  | https://docs.pymc.io/                                        | 贝叶斯建模                           |
| numpyro                | https://github.com/pyro-ppl/numpyro                          | 使用 numpy 进行概率编程              |
| pomegranate            | https://github.com/jmschrei/pomegranate                      | 概率建模                             |
| pmlearn                | [https://github.com/pymc-learn/pymc-learn](https://github.com/pymc-learn/pymc-learn&pmlearn&Probabilistic machine learning.) | 概率机器学习                         |
| arviz                  | https://github.com/arviz-devs/arviz                          | 贝叶斯模型的探索性分析               |
| zhusuan                | https://github.com/thu-ml/zhusuan                            | 贝叶斯深度学习，生成模型             |
| dowhy                  | https://github.com/Microsoft/dowhy                           | 估计因果效应                         |
| edward                 | https://github.com/blei-lab/edward                           | 概率建模、推理和批评                 |
| Pyro                   | https://github.com/pyro-ppl/pyro                             | 深度通用概率编程                     |
| tensorflow probability | https://github.com/tensorflow/probability                    | 深度学习和概率建模                   |
| bambi                  | https://github.com/bambinos/bambi                            | 在PyMC3 之上的高级贝叶斯模型构建接口 |
| neural-tangents        | https://github.com/google/neural-tangents                    | 无限神经网络                         |
| GPyOpt                 | https://github.com/SheffieldML/GPyOpt                        | 使用 GPy 的高斯过程优化              |
| GPflow                 | https://github.com/GPflow/GPflow                             | 高斯过程（Tensorflow）               |
| gpytorch               | https://gpytorch.ai/                                         | 高斯过程（Pytorch）                  |

#### 时间序列模型

| 名称             | 地址                                                  | 说明                                             |
| ---------------- | ----------------------------------------------------- | ------------------------------------------------ |
| statsmodels      | https://github.com/statsmodels/statsmodels            | 时间序列分析                                     |
| kats             | https://github.com/facebookresearch/kats              | Facebook 的时间序列预测库                        |
| prophet          | https://github.com/facebook/prophet                   | Facebook 的时间序列预测库                        |
| neural_prophet   | https://github.com/ourownstory/neural_prophet         | 基于 Pytorch 的时间序列预测                      |
| pyramid          | https://github.com/tgsmith61591/pyramid               | ARIMA的包装                                      |
| pyflux           | https://github.com/RJT1990/pyflux                     | 时间序列预测算法                                 |
| atspy            | https://github.com/firmai/atspy                       | 自动化时间序列模型                               |
| pm-prophet       | https://github.com/luke14free/pm-prophet              | 时间序列预测和分解库                             |
| htsprophet       | https://github.com/CollinRooney12/htsprophet          | 使用 Prophet 进行分层时间序列预测                |
| nupic            | https://github.com/numenta/nupic                      | 用于时间序列预测和异常检测的分层时间记忆 (HTM)   |
| tensorflow       | https://github.com/tensorflow/tensorflow/             | LSTM 等                                          |
| tspreprocess     | https://github.com/MaxBenChrist/tspreprocess          | 预处理：去噪、压缩、重采样                       |
| tsfresh          | https://github.com/blue-yonder/tsfresh                | 时间序列特征工程                                 |
| thunder          | https://github.com/thunder-project/thunder            | 用于加载、处理和分析时间序列数据的数据结构和算法 |
| gatspy           | https://www.astroml.org/gatspy/                       | 天文时间序列的通用工具                           |
| gendis           | https://github.com/IBCNServices/GENDIS                | shapelets                                        |
| tslearn          | https://github.com/rtavenar/tslearn                   | 时间序列聚类和分类                               |
| pastas           | https://pastas.readthedocs.io/en/latest/examples.html | 时间序列的模拟                                   |
| fastdtw          | https://github.com/slaypni/fastdtw                    | 动态时间扭曲距离                                 |
| pydlm            | https://github.com/wwrechard/pydlm                    | 贝叶斯时间序列建模                               |
| PyAF             | https://github.com/antoinecarme/pyaf                  | 自动时间序列预测                                 |
| luminol          | https://github.com/linkedin/luminol                   | Linkedin 的异常检测和相关库                      |
| matrixprofile-ts | https://github.com/target/matrixprofile-ts            | 检测模式和异常                                   |
| stumpy           | https://github.com/TDAmeritrade/stumpy                | 另一个矩阵配置文件库                             |
| obspy            | https://github.com/obspy/obspy                        | 用于地震学的包                                   |
| RobustSTL        | https://github.com/LeeDoYup/RobustSTL                 | 稳健的季节性趋势分解                             |
| seglearn         | https://github.com/dmbee/seglearn                     | 时间序列库                                       |
| pyts             | https://github.com/johannfaouzi/pyts                  | 时间序列转换和分类                               |
| sktime           | https://github.com/alan-turing-institute/sktime       | 基于深度学习处理时间序列                         |
| adtk             | https://github.com/arundo/adtk                        | 时间序列异常检测                                 |
| rocket           | https://github.com/angus924/rocket                    | 使用随机卷积核的时间序列分类                     |
| luminaire        | https://github.com/zillow/luminaire                   | 时间序列的异常检测                               |
| etna             | https://github.com/tinkoff-ai/etna                    | 时间序列库                                       |

#### 生存分析模型

| 名称                 | 地址                                             | 说明                  |
| -------------------- | ------------------------------------------------ | --------------------- |
| lifelines            | https://lifelines.readthedocs.io/en/latest       | 生存分析，Cox PH 回归 |
| scikit-survival      | https://github.com/sebp/scikit-survival          | 生存分析              |
| xgboost              | https://github.com/dmlc/xgboost                  | 生存分析              |
| survivalstan         | https://github.com/hammerlab/survivalstan        | 生存分析              |
| convoys              | https://github.com/better/convoys                | 分析时间滞后的转化    |
| pysurvival           | https://github.com/square/pysurvival             | 生存分析              |
| DeepSurvivalMachines | https://github.com/autonlab/DeepSurvivalMachines | 全参数生存回归        |

#### 图神经网络模型

| 名称              | 地址                                         | 说明                       |
| ----------------- | -------------------------------------------- | -------------------------- |
| ogb               | https://ogb.stanford.edu/                    | 开源基准图模型，基准数据集 |
| networkx          | https://github.com/networkx/networkx         | 图形库                     |
| cugraph           | https://github.com/rapidsai/cugraph          | GPU 上的图形库             |
| pytorch-geometric | https://github.com/rusty1s/pytorch_geometric | 基于图的深度学习的各种方法 |
| dgl               | https://github.com/dmlc/dgl                  | 深度图模型的库             |
| graph_nets        | https://github.com/deepmind/graph_nets       | 在 Tensorflow 中构建图网络 |

#### 多标签模型

| 名称              | 地址                                                   | 说明           |
| ----------------- | ------------------------------------------------------ | -------------- |
| scikit-multilearn | https://github.com/scikit-multilearn/scikit-multilearn | 多标签模型的库 |

#### 噪声标签学习

| 名称     | 地址                                 | 说明                               |
| -------- | ------------------------------------ | ---------------------------------- |
| cleanlab | https://github.com/cleanlab/cleanlab | 用于带有噪声标签的机器学习         |
| doubtlab | https://github.com/koaning/doubtlab  | 帮助在数据集中找到错误或嘈杂的标签 |

#### 在线学习

| 名称     | 地址                                    | 说明                                                        |
| -------- | --------------------------------------- | ----------------------------------------------------------- |
| river    | https://github.com/online-ml/river      | 在线机器学习库                                              |
| Kaggler  | https://github.com/jeongyoonlee/Kaggler | 用于 ETL 和数据分析的轻量级在线机器学习算法和实用程序功能包 |
| onelearn | https://github.com/onelearn/onelearn    | 使用 Python 进行在线学习的小型 Python 包。                  |

#### 主动学习

| 名称  | 地址                                  | 说明         |
| ----- | ------------------------------------- | ------------ |
| modAL | https://github.com/modAL-python/modAL | 主动学习框架 |

#### 度量学习

| 名称                    | 地址                                                         | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| metric-learn            | https://github.com/scikit-learn-contrib/metric-learn         | 包含几种流行的监督和弱监督度量学习算法的高效 Python 实现     |
| pytorch-metric-learning | https://github.com/KevinMusgrave/pytorch-metric-learning     | 深度度量学习的最简单方法，用 PyTorch 编写                    |
| deep_metric_learning    | https://github.com/ronekko/deep_metric_learning              | 几种深度度量学习方法的实现                                   |
| ivis                    | https://bering-ivis.readthedocs.io/en/latest/supervised.html | 监督降维                                                     |
| tensorflow similarity   | https://github.com/tensorflow/similarity                     | 用于相似性学习的 TensorFlow 库，包括自我监督学习、度量学习、相似性学习和对比学习等技术 |

#### 自监督学习

| 名称    | 地址                                      | 说明                             |
| ------- | ----------------------------------------- | -------------------------------- |
| lightly | https://github.com/lightly-ai/lightly     | 用于自我监督学习的计算机视觉框架 |
| vissl   | https://github.com/facebookresearch/vissl | 用于带有图像的 SOTA 自监督学习   |

#### 强化学习

| 名称    | 地址                                            | 说明                                                         |
| ------- | ----------------------------------------------- | ------------------------------------------------------------ |
| RLLib   | https://ray.readthedocs.io/en/latest/rllib.html | 用于强化学习 (RL) 的开源库                                   |
| Horizon | https://github.com/facebookresearch/Horizon/    | Facebook 开发和使用的应用强化学习 (RL) 的开源端到端平台，包含训练流行的深度 RL 算法的工作流，包括数据预处理、特征转换、分布式训练、反事实策略评估和优化服务 |

### 特征工程

#### 数据可视化

| 名称                | 地址                                                         | 说明                                          |
| ------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| cufflinks           | https://github.com/santosjorge/cufflinks                     | 动态可视化库                                  |
| physt               | https://github.com/janpipek/physt                            | 更好的直方图                                  |
| fast-histogram      | https://github.com/astrofrog/fast-histogram                  | 快速直方图                                    |
| matplotlib_venn     | https://github.com/konstantint/matplotlib-venn               | 维恩图                                        |
| joypy               | https://github.com/sbebo/joypy                               | 绘制堆积密度图                                |
| mosaic plots        | https://www.statsmodels.org/dev/generated/statsmodels.graphics.mosaicplot.mosaic.html | 分类变量可视化                                |
| scikit-plot         | https://github.com/reiinakano/scikit-plot                    | 机器学习模型的 ROC 曲线和其他可视化           |
| yellowbrick         | https://github.com/DistrictDataLabs/yellowbrick              | ML  模型的可视化                              |
| bokeh               | https://bokeh.pydata.org/en/latest/                          | 交互式可视化库                                |
| lets-plot           | https://github.com/JetBrains/lets-plot/blob/master/README_PYTHON.md | 绘图库                                        |
| animatplot          | https://github.com/t-makaro/animatplot                       | 动画绘图基于 matplotlib                       |
| plotnine            | https://github.com/has2k1/plotnine                           | 用于 Python 的  ggplot                        |
| altair              | https://altair-viz.github.io/                                | 声明性统计可视化库                            |
| bqplot              | https://github.com/bloomberg/bqplot                          | IPython/Jupyter  Notebooks 的绘图库           |
| hvplot              | https://github.com/pyviz/hvplot                              | 建立在 http://holoviews.org/ 之上的高级绘图库 |
| dtreeviz            | https://github.com/parrt/dtreeviz                            | 决策树可视化和模型解释                        |
| chartify            | https://github.com/spotify/chartify/                         | 生成图表                                      |
| python-ternary      | https://github.com/marcharper/python-ternary                 | 三角图                                        |
| falcon              | https://github.com/uwdata/falcon                             | 大数据的交互式可视化                          |
| hiplot              | https://github.com/facebookresearch/hiplot                   | 高维交互式绘图                                |
| visdom              | https://github.com/fossasia/visdom                           | 实时可视化                                    |
| mpl-scatter-density | https://github.com/astrofrog/mpl-scatter-density             | 散点密度图，替代二维直方图                    |

#### 特征衍生

| 名称              | 地址                                                      | 说明     |
| ----------------- | --------------------------------------------------------- | -------- |
| category_encoders | https://github.com/scikit-learn-contrib/category_encoders | 特征编码 |



#### 特征选择

| 名称                | 地址                                                         | 说明                                     |
| ------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| sklearn             | https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection | 特征选择                                 |
| eli5                | https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html#feature-selection | 使用排列重要性进行特征选择               |
| scikit-feature      | https://github.com/jundongl/scikit-feature                   | 特征选择算法                             |
| stability-selection | https://github.com/scikit-learn-contrib/stability-selection  | 稳定性选择                               |
| scikit-rebate       | https://github.com/EpistasisLab/scikit-rebate                | 基于浮雕的特征选择算法                   |
| scikit-genetic      | https://github.com/manuel-calzolari/sklearn-genetic          | 遗传特征选择                             |
| boruta_py           | https://github.com/scikit-learn-contrib/boruta_py            | 特征选择                                 |
| linselect           | https://github.com/efavdb/linselect                          | 功能选择包                               |
| mlxtend             | https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/ | 详尽的特征选择                           |
| BoostARoota         | https://github.com/chasedehan/BoostARoota                    | Xgboost  特征选择算法                    |
| INVASE              | https://github.com/jsyoon0823/INVASE                         | 使用神经网络的实例变量选择               |
| SubTab              | https://github.com/AstraZeneca/SubTab                        | 用于自我监督表示学习的表格数据的子集特征 |
| mrmr                | https://github.com/smazzanti/mrmr                            | 最大相关性和最小冗余特征选择             |
| feature-selector    | https://github.com/WillKoehrsen/feature-selector             | 特征选择器                               |

#### 表征学习

| 名称                      | 地址                                                         | 说明                                                    |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| sklearn.manifold          | https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold | PCA、t-SNE、MDS、Isomap  等                             |
| sklearn.decomposition     | https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition | PCA、t-SNE、MDS、Isomap  等                             |
| sklearn.random_projection | https://scikit-learn.org/stable/modules/random_projection.html | Johnson-Lindenstrauss  引理、高斯随机投影、稀疏随机投影 |
| prince                    | https://github.com/MaxHalford/prince                         | 降维、因子分析（PCA、MCA、CA、FAMD）                    |
| Faster t-SNE              | https://lvdmaaten.github.io/tsne/                            | MulticoreTSNE                                           |
| umap                      | https://github.com/lmcinnes/umap                             | 统一流形逼近和投影                                      |
| somoclu                   | https://github.com/peterwittek/somoclu                       | 自组织图                                                |
| scikit-tda                | https://github.com/scikit-tda/scikit-tda                     | 拓扑数据分析                                            |
| giotto-tda                | https://github.com/giotto-ai/giotto-tda                      | 拓扑数据分析                                            |
| ivis                      | https://github.com/beringresearch/ivis                       | 使用连体网络降维                                        |
| trimap                    | https://github.com/eamid/trimap                              | 使用三元组进行降维                                      |
| scanpy                    | https://github.com/theislab/scanpy                           |                                                         |
| direpack                  | https://github.com/SvenSerneels/direpack                     | 投影追踪、足够的降维、鲁棒的M估计器                     |
| contrastive               | https://github.com/abidlabs/contrastive                      | 对比  PCA                                               |

#### 异常检测

| 名称      | 地址                                                         | 说明                                   |
| --------- | ------------------------------------------------------------ | -------------------------------------- |
| sklearn   | https://scikit-learn.org/stable/modules/outlier_detection.html | 隔离森林等                             |
| pyod      | https://pyod.readthedocs.io/en/latest/pyod.html              | 异常值检测/异常检测                    |
| eif       | https://github.com/sahandha/eif                              | 扩展隔离林                             |
| luminol   | https://github.com/linkedin/luminol                          | Linkedin  的异常检测和相关库           |
| banpei    | https://github.com/tsurubee/banpei                           | 基于奇异谱变换的异常检测库             |
| telemanom | https://github.com/khundman/telemanom                        | 使用 LSTM 检测多元时间序列数据中的异常 |
| luminaire | https://github.com/zillow/luminaire                          | 时间序列的异常检测                     |

#### 聚类算法

| 名称                    | 地址                                                         | 说明                                        |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------- |
| hdbscan                 | https://github.com/scikit-learn-contrib/hdbscan              | 聚类算法                                    |
| pyclustering            | https://github.com/annoviko/pyclustering                     | 各种聚类算法                                |
| GaussianMixture         | https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html | 使用混合高斯分布的广义 k 均值聚类           |
| nmslib                  | https://github.com/nmslib/nmslib                             | 用于评估 k-NN 方法的相似性搜索库和工具包    |
| buckshotpp              | https://github.com/zjohn77/buckshotpp                        | 抗异常值和可扩展的聚类算法                  |
| merf                    | https://github.com/manifoldai/merf                           | 用于聚类的混合效应随机森林                  |
| tree-SNE                | https://github.com/isaacrob/treesne                          | 基于t-SNE的层次聚类算法                     |
| MiniSom                 | https://github.com/JustGlowing/minisom                       | 自组织地图的纯 Python 实现                  |
| distribution_clustering | https://github.com/EricElmoznino/distribution_clustering     |                                             |
| phenograph              | https://github.com/dpeerlab/phenograph                       | 通过社区检测进行聚类                        |
| FastPG                  | https://github.com/sararselitsky/FastPG                      | 单细胞数据 (RNA) 的聚类，phenograph  的改进 |
| HypHC                   | https://github.com/HazyResearch/HypHC                        | 双曲层次聚类                                |
| BanditPAM               | https://github.com/ThrunGroup/BanditPAM                      | 改进的 k-Medoids 聚类                       |

#### 缺失填补

| 名称      | 地址                                         | 说明         |
| --------- | -------------------------------------------- | ------------ |
| missingpy | https://github.com/epsilon-machine/missingpy | 缺失填补的包 |



#### 自动化特征工程

| 名称               | 地址                                                         | 说明                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| AdaNet             | https://github.com/tensorflow/adanet                         | 基于张量流的自动化机器学习                                   |
| tpot               | https://github.com/EpistasisLab/tpot                         | 自动化机器学习工具，优化机器学习管道                         |
| auto_ml            | https://github.com/ClimbsRocks/auto_ml                       | 用于分析的自动化机器学习                                     |
| auto-sklearn       | https://github.com/automl/auto-sklearn                       | 使用 scikit-learn 进行自动化机器学习                         |
| autokeras          | https://github.com/jhfjhfj1/autokeras                        | 用于深度学习的 AutoML                                        |
| nni                | https://github.com/Microsoft/nni                             | Microsoft  用于神经架构搜索和超参数调整的工具包              |
| automl-gs          | https://github.com/minimaxir/automl-gs                       | 自动化机器学习                                               |
| mljar              | https://github.com/mljar/mljar-supervised                    | 自动化机器学习                                               |
| automl_zero        | https://github.com/google-research/google-research/tree/master/automl_zero | 自动发现可以解决来自 Google 的机器学习任务的计算机程序       |
| AlphaPy            | https://github.com/ScottfreeLLC/AlphaPy                      | 使用 scikit-learn xgboost、LightGBM 等进行自动化机器学习     |
| automlbenchmark    | https://github.com/openml/automlbenc                         | 提供了一个用于评估和比较开源 AutoML 系统的框架。             |
| datacleaner        | https://github.com/rhiever/datacleaner                       | 一种 Python 工具，可自动清理数据集并为分析做好准备。         |
| automl_benchmark   | https://github.com/Ennosigaeon/automl_benchmark              | 评估流行 CASH 和 AutoML 框架的基准                           |
| AutoX              | https://github.com/4paradigm/AutoX                           | AutoX 是一款高效的 automl 工具，主要针对表格数据的数据挖掘比赛。 |
| featuretools       | https://github.com/alteryx/featuretools                      | 用于自动化特征工程的开源 python 库                           |
| EvolutionaryForest | https://github.com/zhenlingcn/EvolutionaryForest             | 基于遗传编程的自动化特征工程的开源python库                   |

### 其它工具

#### 模型调参的工具

| 名称                      | 地址                                                    | 说明                                     |
| ------------------------- | ------------------------------------------------------- | ---------------------------------------- |
| sklearn                   | https://scikit-learn.org/stable/index.html              | GridSearchCV，RandomizedSearchCV         |
| sklearn-deap              | https://github.com/rsteca/sklearn-deap                  | 使用遗传算法进行超参数搜索               |
| hyperopt                  | https://github.com/hyperopt/hyperopt                    | 超参数优化                               |
| hyperopt-sklearn          | https://github.com/hyperopt/hyperopt-sklearn            | Hyperopt  + sklearn                      |
| optuna                    | https://github.com/pfnet/optuna                         | 超参数优化                               |
| skopt                     | https://scikit-optimize.github.io/                      | 用于超参数搜索                           |
| tune                      | https://ray.readthedocs.io/en/latest/tune.html          | 专注于深度学习和深度强化学习的超参数搜索 |
| hypergraph                | https://github.com/aljabr0/hypergraph                   | 全局优化方法和超参数优化                 |
| bbopt                     | https://github.com/evhub/bbopt                          | 黑盒超参数优化                           |
| dragonfly                 | https://github.com/dragonfly/dragonfly                  | 可扩展贝叶斯优化                         |
| botorch                   | https://github.com/pytorch/botorch                      | PyTorch  中的贝叶斯优化                  |
| ax                        | https://github.com/facebook/Ax                          | Facebook 的自适应实验平台                |
| EvolutionaryParameterGrid | https://github.com/zhenlingcn/EvolutionaryParameterGrid | 通用组合优化问题求解器                   |

#### 模型评估的工具

| 名称        | 地址                                                         | 说明                      |
| ----------- | ------------------------------------------------------------ | ------------------------- |
| pycm        | https://github.com/sepandhaghighi/pycm                       | 多类混淆矩阵              |
| pandas-ml   | https://github.com/pandas-ml/pandas-ml                       | 混淆矩阵                  |
| yellowbrick | http://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html | 学习曲线                  |
| pyroc       | https://github.com/noudald/pyroc                             | 接受者操作特征 (ROC) 曲线 |

#### 模型解释的工具

| 名称                    | 地址                                                   | 说明                                                         |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| shap                    | https://github.com/slundberg/shap                      | 解释机器学习模型的预测                                       |
| treeinterpreter         | https://github.com/andosa/treeinterpreter              | 解释 scikit-learn 的决策树和随机森林预测                     |
| lime                    | https://github.com/marcotcr/lime                       | 解释任何机器学习分类器的预测                                 |
| lime_xgboost            | https://github.com/jphall663/lime_xgboost              | 为 XGBoost 创建  LIME                                        |
| eli5                    | https://github.com/TeamHG-Memex/eli5                   | 检查机器学习分类器并解释它们的预测                           |
| lofo-importance         | https://github.com/aerdem4/lofo-importance             | 忽略一项功能的重要性                                         |
| pybreakdown             | https://github.com/MI2DataLab/pyBreakDown              | 生成特征贡献图                                               |
| FairML                  | https://github.com/adebayoj/fairml                     | 模型解释，特征重要性                                         |
| pycebox                 | https://github.com/AustinRochford/PyCEbox              | 个体条件期望图工具箱                                         |
| pdpbox                  | https://github.com/SauceCat/PDPbox                     | 部分依赖图工具箱                                             |
| partial_dependence      | https://github.com/nyuvis/partial_dependence           | 可视化和聚类部分依赖                                         |
| skater                  | https://github.com/datascienceinc/Skater               | 支持模型解释的统一框架                                       |
| anchor                  | https://github.com/marcotcr/anchor                     | 分类器的高精度模型无关解释                                   |
| l2x                     | https://github.com/Jianbo-Lab/L2X                      | 实例特征选择作为模型解释的方法                               |
| contrastive_explanation | https://github.com/MarcelRobeer/ContrastiveExplanation | 对比解释                                                     |
| DrWhy                   | https://github.com/ModelOriented/DrWhy                 | 可解释 AI 的工具集                                           |
| lucid                   | https://github.com/tensorflow/lucid                    | 神经网络可解释性                                             |
| xai                     | https://github.com/EthicalML/XAI                       | 用于机器学习的可解释性工具箱                                 |
| innvestigate            | https://github.com/albermax/innvestigate               | 研究神经网络预测的工具箱                                     |
| dalex                   | https://github.com/pbiecek/DALEX                       | ML  模型说明（R 包）                                         |
| interpretml             | https://github.com/interpretml/interpret               | 拟合可解释模型，解释模型                                     |
| shapash                 | https://github.com/MAIF/shapash                        | 模型可解释性                                                 |
| imodels                 | https://github.com/csinva/imodels                      | 可解释的机器学习包                                           |
| TrustScore              | https://github.com/google/TrustScore                   | 任何经过训练（可能是黑盒）分类器的不确定性度量，它比分类器自己的隐含置信度更有效 |
| dtreeviz                | https://github.com/parrt/dtreeviz                      | 用于决策树可视化和模型解释的 python 库。 目前支持 scikit-learn、XGBoost、Spark MLlib 和 LightGBM 树 |

#### 项目管理的工具

| 名称     | 地址                                  | 说明                                                         |
| -------- | ------------------------------------- | ------------------------------------------------------------ |
| dvc      | https://dvc.org/                      | 大文件的版本控制                                             |
| gigantum | https://github.com/gigantum           |                                                              |
| mlflow   | https://mlflow.org/                   | 机器学习生命周期的开源平台                                   |
| mlmd     | https://github.com/google/ml-metadata | 用于记录和检索与 ML 开发人员和数据科学家工作流相关的元数据的库 |
| modeldb  | https://github.com/VertaAI/modeldb    | 一个开源系统，用于对机器学习模型进行版本控制，包括其成分代码、数据、配置和环境，并在整个模型生命周期中跟踪 ML 元数据。 |
| whylabs  | https://www.rsqrdai.org/              |                                                              |

### 因果推断

| 名称                   | 地址                                                       | 说明                                                         |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| pyCausalFS             | https://github.com/wt-hu/pyCausalFS                        | 用于因果结构学习和分类的基于因果关系的特征选择 Python 库     |
| Causal-Learner         | https://github.com/z-dragonl/Causal-Learner                | 因果结构和马尔可夫毯式学习的工具箱                           |
| CausalFS               | https://github.com/kuiy/CausalFS                           | 因果特征选择和因果（贝叶斯网络）结构学习的开源包（C++版）    |
| CausalDiscoveryToolbox | https://github.com/FenTechSolutions/CausalDiscoveryToolbox | 用于图形和成对设置中的因果推断的软件包。 包括用于图结构恢复和依赖关系的工具 |
| pgmpy                  | https://github.com/pgmpy/pgmpy                             | 用于贝叶斯网络中的学习（结构和参数）、推理（概率和因果）和模拟的 Python 库 |
| causalml               | https://github.com/uber/causalml                           | 提供了一套使用基于最近研究的机器学习算法的提升建模和因果推理方法 ， 它提供了一个标准界面，允许用户根据实验或观察数据估计条件平均治疗效果 (CATE) 或个体治疗效果 (ITE) |
| EconML                 | https://github.com/microsoft/EconML                        | 用于通过机器学习从观察数据中估计异构治疗效果。 该软件包是微软研究院 ALICE 项目的一部分，旨在将最先进的机器学习技术与计量经济学相结合，为复杂的因果推理问题带来自动化 |
| tigramite              | https://github.com/jakobrunge/tigramite                    | 一个用于因果发现的时间序列分析                               |
| causal-learn           | https://github.com/cmu-phil/causal-learn                   | 因果发现包，它实现了经典和最先进的因果发现算法，它是Tetrad 的Python扩展 |
| dagitty                | https://github.com/jtextor/dagitty                         | 结构因果模型/图形因果模型的图形分析                          |
| DoWhy                  | https://github.com/microsoft/dowhy                         | 是因果推理的统一语言，结合了因果图形模型和潜在结果框架，支持显式建模和因果假设测试 |
| WhyNot                 | https://github.com/zykls/whynot                            | 该软件包为动态决策提供了一个实验沙箱，将因果推理和强化学习的工具与具有挑战性的动态环境连接起来，该软件包有助于开发、测试、基准测试和因果推理和顺序决策工具。 |
| JustCause              | https://github.com/inovex/justcause                        | 开发一个框架，允许您以公平公正的方式比较因果推理的方法。     |
| CausalNex              | https://github.com/quantumblacklabs/causalnex              | CausalNex 建立在我们利用贝叶斯网络识别数据中的因果关系的集体经验之上，以便我们可以从分析中开发正确的干预措施 |
| Trustworthy AI         | https://github.com/huawei-noah/trustworthyAI               | 因果结构学习、因果解耦表征学习                               |

### 其它的AWESOME

| 名称                                 | 地址                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| Awesome Adversarial Machine Learning | https://github.com/yenchenlin/awesome-adversarial-machine-learning |
| Awesome AI Booksmarks                | https://github.com/goodrahstar/my-awesome-AI-bookmarks       |
| Awesome AI on Kubernetes             | https://github.com/CognonicLabs/awesome-AI-kubernetes        |
| Awesome Big Data                     | https://github.com/onurakpolat/awesome-bigdata               |
| Awesome Community Detection          | https://github.com/benedekrozemberczki/awesome-community-detection |
| Awesome CSV                          | https://github.com/secretGeek/AwesomeCSV                     |
| Awesome Data Science with Ruby       | https://github.com/arbox/data-science-with-ruby              |
| Awesome Dash                         | https://github.com/ucg8j/awesome-dash                        |
| Awesome Decision Trees               | https://github.com/benedekrozemberczki/awesome-decision-tree-papers |
| Awesome Deep Learning                | https://github.com/ChristosChristofidis/awesome-deep-learning |
| Awesome ETL                          | https://github.com/pawl/awesome-etl                          |
| Awesome Financial Machine Learning   | https://github.com/firmai/financial-machine-learning         |
| Awesome Fraud Detection              | https://github.com/benedekrozemberczki/awesome-fraud-detection-papers |
| Awesome GAN Applications             | https://github.com/nashory/gans-awesome-applications         |
| Awesome Graph Classification         | https://github.com/benedekrozemberczki/awesome-graph-classification |
| Awesome Gradient Boosting            | https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers |
| Awesome Machine Learning             | https://github.com/josephmisiti/awesome-machine-learning#python |
| Awesome Machine Learning Books       | http://matpalm.com/blog/cool_machine_learning_books/"  rel="nofollow |
| Awesome Machine Learning Operations  | https://github.com/EthicalML/awesome-machine-learning-operations |
| Awesome Metric Learning              | https://github.com/kdhht2334/Survey_of_Deep_Metric_Learning  |
| Awesome Monte Carlo Tree Search      | https://github.com/benedekrozemberczki/awesome-monte-carlo-tree-search-papers |
| Awesome Neural Network Visualization | https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network |
| Awesome Online Machine Learning      | https://github.com/MaxHalford/awesome-online-machine-learning |
| Awesome Pipeline                     | https://github.com/pditommaso/awesome-pipeline               |
| Awesome Public APIs                  | https://github.com/public-apis/public-apis                   |
| Awesome Python                       | https://github.com/vinta/awesome-python                      |
| Awesome Python Data Science          | https://github.com/krzjoa/awesome-python-datascience         |
| Awesome Python Data Science          | https://github.com/thomasjpfan/awesome-python-data-science   |
| Awesome Python Data Science          | https://github.com/amitness/toolbox                          |
| Awesome Pytorch                      | https://github.com/bharathgs/Awesome-pytorch-list            |
| Awesome Quantitative Finance         | https://github.com/wilsonfreitas/awesome-quant               |
| Awesome Recommender Systems          | https://github.com/grahamjenson/list_of_recommender_systems  |
| Awesome Semantic Segmentation        | https://github.com/mrgloom/awesome-semantic-segmentation     |
| Awesome Sentence Embedding           | https://github.com/Separius/awesome-sentence-embedding       |
| Awesome Visual Attentions            | https://github.com/MenghaoGuo/Awesome-Vision-Attentions      |
| Awesome Visual Transformer           | https://github.com/dk-liang/Awesome-Visual-Transformer       |

