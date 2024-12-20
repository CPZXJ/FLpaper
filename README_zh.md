
[English](/README.md) | [中文](/README_zh.md)
# 联邦学习攻击与防御必读论文

## 介绍
本仓库提供了关于联邦学习（Federated Learning，FL）的精选论文和教程列表，包括系统教程、现有综述、横向联邦学习、纵向联邦学习、联邦迁移学习、联邦推荐场景、数据投毒攻击、模型投毒攻击、数据投毒防御、模型投毒防御、隐私保护的联邦学习、隐私保护的联邦推荐、联邦推荐中的现有攻击和防御方法和一些开源代码等。

\* 请通过提交 [pull request](https://github.com/CPZXJ/FLpaper/pulls) 并使用以下模版来帮助完善此列表，感谢所有的贡献者！
```markdown
- 作者名字. **论文名称.**  会议名称/期刊名称, 发表年份.
```

## 目录
- [**00-系统教程:**](#系统教程) 包含了许多著名研究者在顶级会议上关于联邦学习的系统教程
- [**01-现有综述:**](#现有综述) 一系列关于联邦学习的全面综述，例如横向联邦学习、纵向联邦学习、联邦迁移学习、数据投毒攻击等
- [**02-横向联邦学习:**](#横向联邦学习) 一些关于横向联邦学习的著名论文
- [**03-纵向联邦学习:**](#纵向联邦学习) 一些关于纵向联邦学习的论文
- [**04-联邦迁移学习:**](#联邦迁移学习) 一些关于联邦迁移学习的论文
- [**05-联邦推荐场景:**](#联邦推荐场景) 一些关于联邦推荐应用场景的论文
- [**06-数据投毒攻击:**](#数据投毒攻击) 重点帮助用户理解数据投毒攻击的实现方式
- [**07-模型投毒攻击:**](#模型投毒攻击) 一些解释模型投毒攻击及其应用的论文
- [**08-数据投毒防御:**](#数据投毒防御) 一些专门针对数据投毒攻击的防御论文
- [**09-模型投毒防御:**](#模型投毒防御) 针对模型投毒攻击的防御措施
- [**10-隐私保护的联邦学习:**](#隐私保护的联邦学习) 联邦学习与隐私保护的相关内容
- [**11-隐私保护的联邦推荐:**](#隐私保护的联邦推荐) 在隐私保护中应用联邦推荐
- [**12-联邦推荐中的现有攻击:**](#联邦推荐中的现有攻击) 总结了针对联邦推荐的现有攻击
- [**13-联邦推荐中的现有防御:**](#联邦推荐中的现有防御) 联邦推荐中的一些现有防御方法
- [**14-一些开源代码:**](#一些开源代码) 提供了一些开源代码，便于更好地理解和应用联邦学习


## 系统教程
- Yang et al.  **Federated Learning.**  SLAIML,  2021.

## 现有综述
- Zhou et al.  **联邦学习研究综述.**  网络与信息安全学报,  2021.
- Liang et al.  **基于联邦学习的推荐系统综述.**  中国科学,  2022.
- Wu et al.  **联邦学习攻击与防御综述.**  大数据,  2022.
- Chen et al.  **联邦学习攻防研究综述.**  计算机科学,  2022.
- Gu et al.  **联邦学习模型安全与隐私研究进展.**  软件学报,  2022.
- Shi et al.  **challenges and approaches for mitigating byzantine attacks in federated learning.**  IEEE ICTSP,  2022.
- Xiao et al.  **联邦学习的隐私保护与安全防御研究综述.**  计算机学报,  2023.
- Gao et al.  **联邦学习系统攻击与防御技术研究综述.**  计算机学报,  2023.
- Sun et al.  **联邦学习拜占庭攻击与防御研究综述.**  网络空间安全科学学报,  2023.
- Yang et al.  **联邦学习与攻防对抗综述.**  信息网络安全,  2023.
- Sun et  
al.**Data_and_Model_Poisoning_Backdoor_Attacks_on_Wireless_Federated_Learning_and_the_Defense_Mechanisms_A_Comprehensive_Survey.**  IEEE CSAT,  2024.
- Chen et al.  **联邦学习中的安全威胁与防御措施综述.**  计算机应用,  2024.
- Zhang et al.  **联邦学习中的攻击手段与防御机制研究综述.**  计算机工程与应用,  2024.

## 横向联邦学习
- Zhang et al.  **Federated Feature Selection for Horizontal Federated Learning in IoT Networks.**  IEEE IoT-J,  2023.
## 纵向联邦学习
- Chen et al.  **面向纵向联邦学习的对抗样本生成算法.**  通信学报,  2023.
- Lai et al.  **VFedAD A Defense Method Based on the Information.**  CIKD CCFB,  2023.
## 联邦迁移学习
- Liu et al.  **A Secure Federated Transfer Learning Framework.**  IEEE IS,  2023.
## 联邦推荐场景
- Chen et al.  **一种基于注意力联邦蒸馏的推荐方法.**  软件学报,  2020.
- Rong et al.  **poisoning deep learning based recommender model in federated learning scenarios.**  IJCAL,  2022.

## 数据投毒攻击
- Xiao et al.  **Adversarial label flips attack on support vector machines.**  ECAL,  2012.
- Rosenfeld et al.  **certified robustness to label-flipping attacks via randomized smoothing.**  ICML,  2020.
- Bagdasaryan et al.  **How to backdoor federated learning .**  ICML,  2020.
- Sun et al.  **data posioning attacks on federated machine learning .**  IEEE ITOJ,  2021.
- Xu et al.  **More is Better(mostly)_on the backdoor attacks in federated learning.**  ACSAC,  2022.
- Rieger et al.  **Deepsight_mitigating backdoor attacks in federated learning through deep model inspection.**  NDSS,  2022.
- Fang et al.  **On the vulnerability of backdoor defenses for federated learning.**  AAAI,  2023.
- Liu et al.  **Poisoning_Attack_based_on_Data_Feature_Selection_in_Federated_Learning.**  IEEE Confluence,  2023.
- Xu et al. **Shadowcast Stealthy Data Poisoning Attacks against Vision-language Model** NeurIPS, 2024.
- Pawelczyk et al. **Machine Unlearning Fails to Remove Data Poisoning Attacks** arXiv, 2024.
- Cotroneo et al. **Vulnerabilities in AI Code Generators Exploring Targeted Data Poisoning Attack** ICPC, 2024.
- Yang et al. **Data Poisoning Attacks Against Multimodal Encoders** PMLR, 2023.
- Zhang et al. **Data_Poisoning_based_Backdoor_Attacks_to_Contrastive_Learning** CVPR, 2024.
- Cina et al. **Machine Learning Security against Data Poisoning Are We There Yet** arXiv, 2024.

## 模型投毒攻击
- Baruch et al.  **a-little-is-enough-circumventing-defenses-for-distributed-learning-Paper.**  NeurIPS,  2019.
- Fang et al.  **local model poisoning attacks to byzantine-robust federated learning.**  usenix,  2020.
- Shejwalkar et al.  **Manipulating the byzantine_Optimizing model poisoning attacks and defense for federated learning.**  NDSS,  2021.
- Cao et al.  **MPAF_Model_Poisoning_Attacks_to_Federated_Learning_Based_on_Fake.**  CVPR,  2022.
- Zhang et al.  **denial-of-service or fine-grained control_towards flexible model poisoning attacks on federated learning.**  arxiv,  2023.
- Zhang et al.  **FLTracer_accurate poisoning attack provenance in federated learning.**  arxiv,  2023.
- Jin et al.  **FedPerturb Covert Poisoning Attack on Federated Learning via Partial Perturbation .**  IEEE ECAL,  2023.
- Yuan et al.  **Model_Poisoning_Attack_In_Federated_Learning_Via_Adversarial_Examples.**  IEEE SCSET,  2023.
- Wei et al.  **covert model poisoning against federated learning_algorithm design and optimization.**  IEEE TDSC,  2023.
- Li et al.  **data-agnostic_model poisoning against federated learning_a graph autoencoder approach.**  IEEE TIFS,  2023.
- Zhang et al.  **Denial-of-Service or Fine-Grained Control_towards FLexible model poisoning attacks on federated learning.**  IJCAI,  2023.
- Jiang et al.  **towards efficient and certified recovery from poisoning attacks in federated learning.**  arxiv,  2024.
- Aghakhani et al. **TROJANPUZZLE Covertly Poisoning Code-Suggestion Models** arXiv, 2024.
- Cong et al. **Test-Time Poisoning Attacks Against Test-Time Adaptiation Models** arXiv, 2023.
- Zhang et al. **LoRec Large Language Model for Robust Sequential Recommendation against Poisoning Attacks** arXiv, 2024.
- Wu et al. **Preference Poisoning Attacks on Reward Model Learning** arXiv, 2024.
- Zou et al. **PoisonedRAG Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models** arXiv, 2024.
- Yuan et al. **Robust Federated Contrastive Recommende System against Model Poisoning Attack** arXiv, 2024.
- Xie et al. **Model Poisoning Attacks to Federated Learning via Multi-Round Consistency** arXiv,2024.
- Zhao et al. **Models Are Codes Towards Measuring Malicious Code Poisoning Attacks on Pre-trained Model Hubs** arXiv, 2024.
- Yan et al. **recess-vaccine-for-federated-learning-proactive-defense-against-model-poisoning-attacks** NeurIPS, 2023.
- Li et al. **Poison Attack and Poison Detection on Deep Source Code Processing Models** TOSEM, 2024.
- Yang et al. **robust-contrastive-language-image-pretraining-against-data-poisoning-and-backdoor-attacks** NeurIPS, 2023.

## 数据投毒防御
- Shen et al.  **AUROR：Defending Against Poisoning Attacks in Collaborative Deep Learning Systems.**  ACSAC,  2016.
- Fung et al.  **Mitigating Sybils in Federated Learning Poisoning.**  arxiv,  2018.
- Tolpegin et al.  **Data Poisoning Attacks Against Federated Learning Systems.**  ESORICS,  2020.
- Fung et al.  **The Limitations of Federated learning.**  RAID,  2020.
- Awan et al.  **CONTRA ：defending against Poisoning Attacks in Federated Learning.**  ESORICS,  2021.
- Chen et al.  **De-Pois_An_Attack-Agnostic_Defense_against_Data_Poisoning_Attacks.**  IEEE TIFS,  2021.
- Jebreel et al.  **Defending against the label-flipping attack in federated learning.**  arxiv,  2022.
- Gupta et al.  **Long-ShortHistoryof Gradient is All You Need.**  ESORICS,  2022.
- wu et al.  **Toward_Cleansing_Backdoored_Neural_Networks_in_Federated_Learning.**  ICDCS,  2022.
- Nguyen et al.  **FLAME_Taming backdoors in federated learning.**  USENIX,  2022.
- Rezaei et al.  **Run-off Election_improved Provable Defense against data poisoning attacks.**  arxiv,  2023.
- Jiang et al.  **Data Quality Detection Mechanism Against Label Flipping Attacks in Federated Learning.**  IEEE TIFS,  2023.
- Jebreel et al.  **FL_Defender：Combating targeted attacks in federated learning.**  KBS,  2023.
-Ovi et al.  **Confident federated learning to tackle label flipped data poisoning attacks.**  SPIE DCS,  2023.
- Jebreel et al.  **LFighter_defending against the label-flipping attack in federated learning.**  NN,  2024.
- Li et al. **Defend Data Poisoning Attacks on Voice Authentication** arXiv, 2023.
- Zhao et al. **Defending Against Weight-Poisoning Backdoor Attacks for Parameter-Efficient Fine-Tuning** arXiv, 2024.
- Hong et al. **Diffusion denoising as a certified defense against clean-label poisoning** arXiv, 2024.
- Gaspari et al. **Have you poisoned my data defending neural networks against data poisoning** arXiv, 2024.
- Galanis et al. **Defending against data poisoning attacks in federated learning via user elimination** arXiv, 2024.
- Sun et al. **a gan-based data poisoning attack against federated learning systems and its countermeasure** arXiv, 2024.
- Bhat et al. **PUREGEN universal data purification for train-time ppoison defense via generative model dynamics** arXiv, 2024.
- Wang et al. **Defending against sophisticated poisoning attacks with rl-based aggregation in fedaerated learning** arXiv, 2024.
- Lai et al. **VFedAD a defense method based on the information mechanism behind the vertical federated data poisoning attack** CIKM, 2023.
- Yang et al. **Breaking state-od-the-art poisoning defenses to federated learning an optimization-based attack framework** CIKM,2024.
- Li et al. **BIC-Based mixture model defense against data poisoning attacks on classifiers a comprehensive study** TKDE,2024.
- Badjie et al. **Denoising Autoencoder-Based Defensive Distillation as an Adversarial Robustness Algorithm Against Data Poisoning Attacks** sigada,2024.
- Huang et al. **LDPGuard_Defenses_Against_Data_Poisoning_Attacks_to_Local_Differential_Privacy_Protocols** TKDE,2024.
- Yan et al. **recess-vaccine-for-federated-learning-proactive-defense-against-model-poisoning-attacks** NeurIPS, 2023.
- Wang et al. **temporal-robustness-against-data-poisoning** NeurIPS, 2023.
- Rezaei et al. **Run-off election improved provable defense against data poisoning attacks** PMLR, 2023.

## 模型投毒防御
- McMahan et al.  **communication-efficient learning of deep networks from decentralized data.** MLR, 2016.
- Blanchard et al.  **machine-learning-with-adversaries-byzantine-tolerant-gradient-descent-Paper.** NIPS, 2017.
- Muñoz-González et al.  **Towards poisoning of deep learning algorithms with back-gradient optimization.** arxiv, 2017.
- Yin et al.  **Byzantine-Robust Distributed Learning_towards optimal stastistical rates.** ICML, 2018.
- Mhamdi et al.  **the hidden vulnerability of distributed learning in byzantium.** ICML, 2018.
- Sun et al.  **can you really backdoor federated learning.** arxiv, 2019.
- Baruch et al.  **A little Is Enough_circumventing defense for distributed learning.** NIPS, 2019.
- Bhagoji et al.  **analyzing federated learning through an adversarial lens.** ICML, 2019.
- Li et al.  **abnormal client behavior detection in federated learning.** arxiv, 2019.
- Cao et al.  **Distributed_Gradient_Descent_Algorithm_Robust_to_an_Arbitrary_Number_of_Byzantine_Attackers.** IEEE TOSP, 2019.
- Xia et al.  **FABA_An algorithm for fast aggregation against byzantine attacks in distributed neural networks.** IJCAI, 2019.
- Pillutla et al.  **Robust Aggregation for Federated Learning.** arxiv, 2019.
- Li et al.  **RSAByzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets.** AAAI, 2019.
- Liu et al.  **communication-efficient federated learning for anomaly detection in industrial internet of things.** IEEE GB, 2020.
- Sun et al.  **Data poisoning attacks on federated machine learning.** arxiv, 2020.
- Li et al.  **learning to detect malicious clients for  robust federated learning.** arxiv, 2020.
- Sun et al.  **fl-wbc-enhancing-robustness-against-model-poisoning-attacks-in-federated-learning-from-a-client-perspective-Paper (1).** NeurIPS, 2021.
- Cao et al.  **FLTRUST ：Byzantine-robust federated learning via trust bootstrapping.** NDSS, 2021.
- Shejwalkar et al.  **Manipulating the byzantine_optimizing model poisoning attacks and defenses for federated learning.** NDSS, 2021.
- Xu et al.  **Byzantine-robost Federated learning through Collaborative Malicious Gradient Filtering.** ICDCS, 2022.
- Wang et al.  **Federated unlearning via class-discriminative pruning.** WWW, 2022.
- Cao et al.  **FLCert_Provably_Secure_Federated_Learning_Against_Poisoning_Attacks.** IEEE TIFS, 2022.
- Zhang et al.  **FLDetector_defending federated learning against model poisoning attacks via detecting malicious clients.** KDD, 2022.
- Weng et al.  **Privacy-Preserving_Federated_Learning_based_on_Differential_Privacy_and_Momentum_Gradient_Descent.** IJCNN, 2022.
- Shejwalkar et al.  **Back to the Drawing Board： A critical Evaluation of poisoning attacks on production federated learning.** IEEE-SP, 2022.
- Panda et al.  **SparseFed_mitigating model poisoning attacks in federated learning with sparsification.** AISTATS, 2022.
- Gong et al.  **agramplifier_defending federated learning agaginst poisoning attacks through local update amplification.** IEEE TIFS, 2023.
- Yin et al.  **Defending_Against_Data_Poisoning_Attack_in_Federated_Learning_With_Non-IID_Data.** IEEE TOCSS, 2023.
- Aloran et al.  **Defending_Federated_Learning_Against_Model_Poisoning_Attacks.** IEEE BigData, 2023.
- Yan et al.  **DeFL_defending against model poisoning attacks in federated learning via critical learning periods awareness.** AAAI, 2023.
- Lu et al.  **Depriving_the_Survival_Space_of_Adversaries_Against_Poisoned_Gradients_in_Federated_Learning.** IEEE TIFS, 2023.
- Mozaffari et al.  **every vote count_ranking-based training of federated learning to resist poisoning attacks.** usenix, 2023.
- Guo et al.  **FAST_Adopting_Federated_Unlearning_to_Eliminating_Malicious_Terminals_at_Server_Side.** IEEE TNSE, 2023.
- Zhao et al.  **FedCom_Byzantine-Robust_Federated_Learning_Using_Data_Commitment.** IEEE ICC, 2023.
- Chelli et al.  **FedGuard_Selective_Parameter_Aggregation_for_Poisoning_Attack_Mitigation_in_Federated_Learning.** IEEE CLUSTER, 2023.
- Zhang et al.  **FedRecovery_Differentially_Private_Machine_Unlearning_for_Federated_Learning_Frameworks.** IEEE TIFS, 2023.
-  Cao et al.  **FedRecover_Recovering_from_Poisoning_Attacks_in_Federated_Learning_using_Historical_Information.** IEEE SP, 2023.
- Ebron Jr. et al.  **FedTruth_byzantine_robust and backdoor-resilient federated learning framework.** arxiv, 2023.
- Liu et al.  **FLOW_A_Robust_Federated_Learning_Framework_to_Defend_Against_Model_Poisoning_Attacks_in_IoTs.** IEEE IOTJ, 2023.
- Han et al.  **kick bad guys out zero-knowledge-proof-based anomaly detection in federated learning.** arxiv, 2023.
- Zhu et al.  **leadFL_Client self-defense against model poisoning in federated learning.** PMLR, 2023.
- Li et al.  **loMar_a local defense against poisoning attack on federated learning .** IEEE TDSC, 2023.
- Krauß et al.  **MESAS_poisoning defense for federated learning resilient against adaptive attacks.** CCS, 2023.
- Yan et al.  **reces vaccine for federated learning_proactive defense against model poisoning attacks.** arxiv, 2023.
- Yaldiz et al.  **secure federated learning against modek=l poisonong attacks via client filtering.** ICLR, 2023.
- Zhao et al.  **Semisupervised_Federated-Learning-Based_Intrusion_Detection_Method_for_Internet_of_Things.** IEEE IOTJ, 2023.
- Shen et al.  **SPEFL_Efficient_Security_and_Privacy_Enhanced_Federated_Learning_Against_Poisoning_Attacks.** IEEE IOTJ, 2023.
- Ma et al.  **面向Non-IID数据的拜占庭鲁棒联邦学习.** 通信学报, 2023.
- Yang et al.  **A_Robust_and_Efficient_Federated_Learning_Algorithm_Against_Adaptive_Model_Poisoning_Attacks.** IEEE ITOJ, 2024.
- Dong et al.  **FADngs_Federated_Learning_for_Anomaly_Detection.** IEEE TNNLS, 2024.
- Han et al.  **kick bad guys out_zero-knowlege-proof-based anomaly detection in federated learning.** ICLR, 2024.
- Yang et al.  **RoseAgg_Robust_Defense_against_Targeted_Collusion_Attacks_in_Federated_Learning.** IEEE TIFS, 2024.
- Kasyap et al.  **Sine_Similarity_is_not_enough_for_mitigating_Local_Model_Poisoning_Attacks_in_Federated_Learning.** IEEE TDSC, 2024.
- Huang et al.  **VPPFL_A verifiable privacy- preserving federated learning scheme against poisoning attacks.** CS, 2024.
- Kabir et al. **FLShield a validation based federated learning framework to defend against poisoning attacks** arXiv, 2023.
- Han et al. **FedMID a data-free method for using intermediate outputs as a defense mechanism against poisoning attacks in federated learning** arXiv, 2024.
- Zhao et al. **Models are codes towards measuring malicious code poisoning attacks on pre-trained model hubs** arXiv,2024.
- Xie et al. **FedREDefense defending against model poisoning attacks for federated learning using model update reconstruction error** PMLR, 2024.
- Chen et al. **Tutorial toward robust deep learning against poisoning attacks** TECS, 2023.
- Krau et al. **MESAS poisoning defense for federated learning resilient against adaptive attackers** CCS,2023.
- Sharma et al. **FLAIR defense against model poisoning attack in federated learning** , 2023.
- Zheng et al. **detecting poisoning attacks on federated learning using gradient-weighted class activation mapping** WWW, 2024.
- Yang et al. **A_Robust_and_Efficient_Federated_Learning_Algorithm_against_Adaptive_Model_Poisoning_Attacks** IoT, 2024.
- Zhou et al. **Inversion-Guided_Defense_Detecting_Model_Stealing_Attacks_by_Output_Inverting** TIFS, 2024.
- Wang et al. **defending-against-data-free-model-extraction-by-distributionally-robust-defensive-training** NeurIPS, 2023.
- Zhang et al. **Poisoning-Free_Defense_Against_Black-Box_Model_Extraction** ICASSP, 2024
- Zhu et al. **LeadFL client self-defense against model poisoning in federated learning** PMLR, 2024
- Zhang et al. **Poisoning Free Defense Against Black Box Model Extraction** ICASSP, 2024.
- Yan et al. **DeFL defending against model poisoning attacks in federated learning via critical learning periods awareness** AAAI, 2023.
- Zhu et al. **detection and defense of unlearnable examples** AAAI, 2024.

## 隐私保护的联邦学习
- Liu et al.  **联邦学习中的隐私保护技术.**  软件学报,  2021.

## 隐私保护的联邦推荐
- Zhang et al.  **基于隐私保护的联邦推荐算法综述_张洪磊.**  自动化学报,  2022.
- Xue et al.  **Advanced_Privacy-Preserving_Federated_Relationship_Recommendation.**  IEEE ICAICA,  2023.

## 联邦推荐中的现有攻击
- Chen et al.  **Robust Federated Recommendation System.**  arxiv,  2020.
- Huang et al.  **data poisoning attacks to deep learning based recommender systems.**  NDSS,  2021.
- Rong et al.  **FedRecAttack Model Poisoning Attack to Federated Recommendation.**  IEEE ICDE,  2022.
- Zhang et al.  **PipAttack Poisoning Federated Recommender Systems for.**  arxiv,  2022.
- Wu et al.  **Fedattack_effectivate and covert poisoning attack on federated recommendation via hard sampling.**  kdd,  2022.
- Yu et al.  **Untargeted attack against federated recommendation systems via poisonous item embeddings and the defense.**  AAAI,  2023.

## 联邦推荐中的现有防御
- Sandeepa et al.  **Rec-Def_A_Recommendation-Based_Defence_Mechanism_for_Privacy_Preservation_in_Federated_Learning_Systems.**  IEEE TCE,  2024.

## 一些开源代码
- [**attacking-distributed-learning**](https://github.com/moranant/attacking_distributed_learning.git)

---

| 日期      |  2017年前 | 2017-2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|-----------|-------------|-----------|------|------|------|------|------|
| 论文数量    |      3      |     14    |  10  |   11 |  24  |  60  |  46  |
|   总计     |                           168                              |
