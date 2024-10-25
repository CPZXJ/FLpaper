# Must-read papers on Federated Learning attack and defense  
This repository provides a curated list of papers and tutorials on federated learning (FL), including systematic tutorials, existing reviews, horizontal federated learning, vertical federated learning, federated transfer learning, federated recommendation scenarios, data poisoning attacks, model poisoning attacks, backdoor attacks, data poisoning defense, model poisoning defense, backdoor defense, privacy-preserving federated learning, privacy-preserving federated recommendation, existing attacks in federated recommendation, existing defenses in federated recommendation, some open source code, etc.  

===========================================================================

### [00-Tutorials:](#Tutorials)   contain so many tutorials on federated learning given by prominent researchers at many top-tier conferences
### [01-Reviews:](#Reviews) a set of comprehensive surveys about federated learning，such as horizontal federated learning，vertical federated learning，federated transfer learning，data poisoning attacks and so on
### [02-Horizontal FL:](#Horizontal-Federated-Learning) a set of famous papers on horizontal federated learning
### [03-Vertical FL:](#Vertical-Federated-Learning) several papers on vertical federated learning
### [04-FTL:](#Federated-Transfer-Learning) some papers on federated transfer learning
### [05-FR Scenarios:](#Federated-Recommendation-Scenarios) a set of papers on scenarios applied federated recommendation 
### [06-Data PA:](#Data-Poisoning-Attacks) it focus on helping users understand how a data poisoning attack is implemented
### [07-Model PA:](#Model-Poisoning-Attacks) some papers explaining what is a model poisoning attack and its applications
### [08-BA:](#Backdoor-Attacks) contain so many papers on backdoor attacks
### [09-Data PD:](#Data-Poisoning-Defense) some papers specifically dealing with data poisoning attacks
### [10-Model PD:](#Model-Poisoning-Defense) defenses against model poisoning attacks
### [11-BD:](#Backdoor-Defense)some ways to implement backdoor defense
### [12-PPFL:](#Privacy\-preserving-Federated-Learning) links between federal learning and privacy protection
### [13-PPFR:](#privacy\-preserving-Federated-Recommendation) applying federated recommendation to privacy-preserving
### [14-Existing attacks in FR:](#Existing-Attacks-in-Federated-Recommendation) summary of existing attacks against the federated recommendation
### [15-Existing defenses in FR:](#Existing-Defenses-in-Federated-Recommendation) some existing defenses in federated recommendation
### [16-Some open source code:](#Some-Open-Source-Code) providing users with some open source codes to better understand and apply federated learning

===========================================================================

\* Please help to contribute this list by adding [pull request](https://github.com/CPZXJ/FLpaper/pulls) with the template below.
```markdown
* Author Name et al. **Paper Name.**  Conference/Journal, Year.
```
===========================================================================

## Tutorials
- Yang et al.  **Federated Learning.**  SLAIML,  2021.

## Reviews
- Zhou et al.  **联邦学习研究综述.**  网络与信息安全学报,  2021.
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

## Horizontal Federated Learning
- Zhang et al.  **Federated Feature Selection for Horizontal Federated Learning in IoT Networks.**  IEEE IoT-J,  2023.
## Vertical Federated Learning
- Chen et al.  **面向纵向联邦学习的对抗样本生成算法.**  通信学报,  2023.
## Federated Transfer Learning
- Liu et al.  **A Secure Federated Transfer Learning Framework.**  IEEE IS,  2023.
## Federated Recommendation Scenarios
- Chen et al.  **一种基于注意力联邦蒸馏的推荐方法.**  软件学报,  2020.

## Data Poisoning Attacks
- Xiao et al.  **Adversarial label flips attack on support vector machines.**  ECAL,  2012.
- Rosenfeld et al.  **certified robustness to label-flipping attacks via randomized smoothing.**  ICML,  2020.
- Bagdasaryan et al.  **How to backdoor federated learning .**  ICML,  2020.
- Sun et al.  **data posioning attacks on federated machine learning .**  IEEE ITOJ,  2021.
- Xu et al.  **More is Better(mostly)_on the backdoor attacks in federated learning.**  ACSAC,  2022.
- Rieger et al.  **Deepsight_mitigating backdoor attacks in federated learning through deep model inspection.**  NDSS,  2022.
- Fang et al.  **On the vulnerability of backdoor defenses for federated learning.**  AAAI,  2023.
- Liu et al.  **Poisoning_Attack_based_on_Data_Feature_Selection_in_Federated_Learning.**  IEEE Confluence,  2023.

## Model Poisoning Attacks
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

## Backdoor Attacks
- Huang et al.  **Efficient_any-Target_Backdoor_Attack_with_Pseudo_Poisoned_Samples.**  IEEE ICIP,  2023.

## Data Poisoning Defense
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

## Model Poisoning Defense


## Backdoor Defense
- Shamshiri et al.  **Defense_Method_Challenges_Against_Backdoor_Attacks_in_Neural_Networks.**  IEEE ICAIIC,  2024.

## Privacy-preserving Federated Learning
- Liu et al.  **联邦学习中的隐私保护技术.**  软件学报,  2021.

## Privacy-preserving Federated Recommendation
- Xue et al.  **Advanced_Privacy-Preserving_Federated_Relationship_Recommendation.**  IEEE ICAICA,  2023.

## Existing Attacks in Federated Recommendation
- Rong et al.  **FedRecAttack Model Poisoning Attack to Federated Recommendation.**  IEEE ICDE,  2022.

## Existing Defenses in Federated Recommendation
- Sandeepa et al.  **Rec-Def_A_Recommendation-Based_Defence_Mechanism_for_Privacy_Preservation_in_Federated_Learning_Systems.**  IEEE TCE,  2024.

## Some Open Source Code

===========================================================================

| Data      | 2017-2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|-----------|-----------|------|------|------|------|------|
| PaperN    |           |      |      |      |      |      |



