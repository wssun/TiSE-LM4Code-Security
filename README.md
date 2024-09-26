[//]: # (# ⚔🛡 Security Threats and Protections for Neural Code Models &#40;NCMs&#41;)
# ⚔🛡 Security of Neural Code Models

This repository provides a summary of recent advancements in the security landscape surrounding Neural Code Models, including backdoor, adversarial attacks, corresponding defenses and so on.

**NOTE: We collect the original code from the papers and the code we have reproduced. While, our reproduced code is not guaranteed to be fully accurate and is for reference only. For specific issues, please consult the original authors.**

[//]: # (The bolded article is published by our Nanjing University ISE Lab.)

[![arXiv](https://img.shields.io/badge/xxxx.svg)](https://arxiv.org/abs/xxxx)
![GitHub stars](https://img.shields.io/github/stars/wssun/xxxxx?color=yellow&label=Stars)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[//]: # ([![Awesome]&#40;./figures/logo.svg&#41;]&#40;https://github.com/xxxx&#41;)

## Overview

Neural Code Models (NCMs) are advanced deep learning models that excel in programming language understanding and generation.
NCMs have achieved impressive results across various code intelligence tasks, such as code generation, code summarization, vulnerability/bug detection, and so on. 
However, with the growing use of NCMs in sensitive applications, they have become a prime target for security attacks, which exploit the vulnerabilities inherent in machine learning models. This repository organizes the current knowledge on **Security Threats** and **Defense Strategies** for NCMs.

## Table of Contents

[//]: # (- [📃Survey]&#40;#survey&#41;)
- [📃Survey](#surveies)
- [⚔Security Threats](#security-threats)
  - [Backdoor Attacks](#backdoor-attacks)
  - [Adversarial Attacks](#adversarial-attacks)
  - [Other Attacks](#other-attacks)
- [🛡Defensive Strategies](#defensive-strategies)
  - [Backdoor Defense](#backdoor-defense)
  - [Adversarial Defense](#adversarial-defense)
- [Citation](#citation)

**NOTE: Our paper is labeled with 🚩.**


[//]: # (## Introduction)

[//]: # (Our survey focuses on analyzing **backdoor attacks** and **adversarial attacks** in NCMs, categorizing existing research into different attack vectors and examining their implications on security in software engineering tasks. The review covers both **experimental frameworks** and the **metrics** used for evaluating security risks and attack performance on NCMs.)

[//]: # ()
[//]: # (### Key Contributions:)

[//]: # (- A detailed categorization of attacks on **Neural Code Models &#40;NCMs&#41;**.)

[//]: # (- Review of **experimental setups**, including datasets and metrics used in NCM security research.)

[//]: # (- Insights into the challenges posed by these attacks and opportunities for **future research**.)

## 📃Survey
[//]: # (The threats discussed in the survey are divided into three main categories:)
The survey analyzes security threats to NCMs, categorizing existing attack types such as backdoor and adversarial attacks, and explores their implications for code intelligence tasks.

| Year | Conf./Jour. | Paper                                                                                                      |   
|------|-------------|------------------------------------------------------------------------------------------------------------|
| 2023 | CoRR        | A Survey of Trojans in Neural Models of Source Code: Taxonomy and Techniques.                              |
| 2024 | CoRR        | Robustness, Security, Privacy, Explainability, Efficiency, and Usability of Large Language Models for Code. | 
| 2024 |《软件学报》  | [深度代码模型安全综述](./papers_cn/2024-软件学报-深度代码模型安全综述.pdf)   🚩                                                    |      


## ⚔Security Threats
[//]: # (The threats discussed in the survey are divided into three main categories:)
According to the document, security threats in NCMs are mainly classified into two categories: backdoor attacks and adversarial attacks. Backdoor attacks occur during the training phase, where attackers implant hidden backdoors in the model, allowing it to function normally on benign inputs but behave maliciously when triggered by specific patterns. In contrast, adversarial attacks happen during the testing phase, where carefully crafted perturbations are added to the input, causing the model to make incorrect predictions with high confidence while remaining undetectable to humans.

<img src="./figures/overview.png" alt="An overview of attacks in NCMs." width="600"/>

### Backdoor Attacks
Backdoor attacks inject malicious behavior into the model during training, allowing the attacker to trigger it at inference time using specific triggers:
- **Data poisoning attacks**: Slight changes to the training data that cause backdoor behavior.

| Year | Conf./Jour.     | Paper                                                                             | Code Reporisty                                                                                           | Reproduced Repository |    
|------|-----------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-----------------------|
| 2021 | USENIX Security | Explanation-Guided Backdoor Poisoning Attacks Against Malware Classifiers.        | [![Octocat](./figures/github.svg)](https://github.com/ClonedOne/MalwareBackdoors)                        |                       |
| 2021 | USENIX Security | You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion.         |                                                                                                          |                       |
| 2022 | ICPR            | Backdoors in Neural Models of Source Code.                                        | [![Octocat](./figures/github.svg)](https://github.com/tech-srl/code2seq)                                 |                       |
| 2022 | FSE             | You see what I want you to see: poisoning vulnerabilities in neural code search.  | [![Octocat](./figures/github.svg)](https://github.com/CGCL-codes/naturalcc)                              |                       |
| 2023 | ICPC            | Vulnerabilities in AI Code Generators: Exploring Targeted Data Poisoning Attacks. | [![Octocat](./figures/github.svg)](https://github.com/dessertlab/Targeted-Data-Poisoning-Attacks)        |                       |
| 2023 | ACL             | Backdooring Neural Code Search. 🚩                                                | [![Octocat](./figures/github.svg)](https://github.com/wssun/BADCODE)                                     |                       |
| 2024 | TSE             | Stealthy backdoor attack for code models.                                         | [![Octocat](./figures/github.svg)](https://github.com/yangzhou6666/adversarial-backdoor-for-code-models) |                       |
| 2024 | SP              | Trojanpuzzle: Covertly poisoning code-suggestion models.                          | [![Octocat](./figures/github.svg)](https://github.com/microsoft/CodeGenerationPoisoning)                 |                       |
| 2024 | TOSEM           | Poison attack and poison detection on deep source code processing models.         | [![Octocat](./figures/github.svg)](https://github.com/LJ2lijia/CodeDetector)                             |                       |


- **Model poisoning attacks**: Changes that do not alter the functionality of the code but trick the model.

| Year | Conf./Jour.       | Paper                                                                                            | Code Reporisty                                                                 | Reproduced Repository |  
|------|-------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|-----------------------|
| 2021 | USENIX Security   | You Autocomplete Me: Poisoning Vulnerabilities in Neural Code Completion.                        |                                                                                |                       |
| 2023 | CoRR              | BadCS: A Backdoor Attack Framework for Code search.                                              |                                                                                |                       |
| 2023 | ACL               | Multi-target Backdoor Attacks for Code Pre-trained Models.                                       | [![Octocat](./figures/github.svg)](https://github.com/Lyz1213/Backdoored_PPLM) |                       |
| 2023 | USENIX Security   | PELICAN: Exploiting Backdoors of Naturally Trained Deep Learning Models In Binary Code Analysis. | [![Octocat](./figures/github.svg)](https://github.com/ZhangZhuoSJTU/Pelican)   |                       |



### Adversarial Attacks
These attacks manipulate the input data to deceive the model into making incorrect predictions. Including two categories:
- **White-box attacks**: Attackers have complete knowledge of the target model, including model structure, weight parameters, and training data.

| Year | Conf./Jour. | Paper                                                                                            | Code Reporisty                                                                                | Reproduced Repository |   
|------|-------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------|
| 2018 | CoRR        | Adversarial Binaries for Authorship Identification.                                              |                                                                                               |                       |
| 2020 | OOPSLA      | Adversarial examples for models of code.                                                         | [![Octocat](./figures/github.svg)](https://github.com/tech-srl/adversarial-examples)          |                       |
| 2020 | ICML        | Adversarial robustness for code.                                                                 | [![Octocat](./figures/github.svg)](https://github.com/eth-sri/robust-code)                    |                       |
| 2021 | ICLR        | Generating Adversarial Computer Programs using Optimized Obfuscations.                           | [![Octocat](./figures/github.svg)](https://github.com/ALFA-group/adversarial-code-generation) |                       |
| 2022 | TOSEM       | Towards Robustness of Deep Program Processing Models - Detection, Estimation, and Enhancement.   | [![Octocat](./figures/github.svg)](https://github.com/SEKE-Adversary/CARROT)                  |                       |
| 2022 | ICECCS      | Generating Adversarial Source Programs Using Important Tokens-based Structural Transformations.  |                                                                                               |                       | 
| 2023 | CoRR        | Adversarial Attacks against Binary Similarity Systems.                                           |                                                                                               |                       |
| 2023 | SANER       | How Robust Is a Large Pre-trained Language Model for Code Generation𝑓 A Case on Attacking GPT2. |                                                                                               |                       |

- **Black-box attacks**: Adversaries attackers can only generate adversarial examples by obtaining limited model outputs through model queries.

| Year | Conf./Jour.     | Paper                                                                                                              | Code Reporisty                                                                                  | Reproduced Repository                                            |      
|------|-----------------|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| 2019 | USENIX Security | Misleading Authorship Attribution of Source Code using Adversarial Learning.                                       |                                                                                                 |                                                                  |
| 2019 | CODASPY         | Adversarial authorship attribution in open-source projects.                                                        |                                                                                                 |                                                                  |
| 2020 | CoRR            | STRATA: simple, gradient-free attacks for models of code.                                                          |                                                                                                 |                                                                  |
| 2020 | AAAI            | Generating Adversarial Examples for Holding Robustness of Source Code Processing Models.                           | [![Octocat](./figures/github.svg)](https://github.com/Metropolis-Hastings-Modifier/MHM)         |                                                                  |
| 2021 | TIFS            | A practical black-box attack on source code authorship identification classifiers.                                 |                                                                                                 |                                                                  |
| 2021 | ICST            | A Search-Based Testing Framework for Deep Neural Networks of Source Code Embedding.                                | [![Octocat](./figures/github.svg)](https://github.com/MaryamVP/Guided-Mutation-ICST-2021)       |                                                                  |
| 2021 | QRS             | Generating adversarial examples of source code classification models via q-learning-based markov decision process. |                                                                                                 |                                                                  |
| 2021 | GECCO           | Deceiving neural source code classifiers: finding adversarial examples with grammatical evolution.                 | [![Octocat](./figures/github.svg)](https://github.com/Martisal/adversarialGE)                   |                                                                  |
| 2021 | CoRR            | On adversarial robustness of synthetic code generation.                                                            | [![Octocat](./figures/github.svg)](https://github.com/mtensor/neural_sketch)                    |                                                                  |
| 2022 | TOSEM           | Adversarial Robustness of Deep Code Comment Generation.                                                            | [![Octocat](./figures/github.svg)](https://github.com/zhangxq-1/ACCENT-repository)              |                                                                  |
| 2022 | ICSE            | Natural Attack for Pre-trained Models of Code.                                                                     | [![Octocat](./figures/github.svg)](https://github.com/soarsmu/attack-pretrain-models-of-code)   | [![Localcat](./figures/octocat.png)](./adversarial_attack/ALTER) |
| 2022 | ICSE            | RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation.                      | [![Octocat](./figures/github.svg)](https://github.com/RoPGen/RoPGen)                            |                                                                  |
| 2022 | EMNLP           | TABS: Efficient Textual Adversarial Attack for Pre-trained NL Code Model Using Semantic Beam Search.               |                                                                                                 |                                                                  |
| 2023 | AAAI            | CodeAttack: Code-Based Adversarial Attacks for Pre-trained Programming Language Models.                            | [![Octocat](./figures/github.svg)](https://github.com/reddy-lab-code-research/CodeAttack)       |                                                                  |
| 2023 | PACM PL         | Discrete Adversarial Attack to Models of Code.                                                                     |                                                                                                 |                                                                  |
| 2023 | CoRR            | Adversarial Attacks on Code Models with Discriminative Graph Patterns.                                             |                                                                                                 |                                                                  |
| 2023 | Electronics     | AdVulCode: Generating Adversarial Vulnerable Code against Deep Learning-Based Vulnerability Detectors.             |                                                                                                 |                                                                  |
| 2023 | ACL             | DIP: Dead code Insertion based Black-box Attack for Programming Language Model.                                    |                                                                                                 |                                                                  |
| 2023 | CoRR            | A Black-Box Attack on Code Models via Representation Nearest Neighbor Search.                                      | [![Octocat](./figures/github.svg)](https://github.com/18682922316/RNNS-for-code-attack)         |                                                                  |
| 2023 | CoRR            | SHIELD: Thwarting Code Authorship Attribution.                                                                     |                                                                                                 |                                                                  |
| 2023 | ASE             | Code difference guided adversarial example generation for deep code models.                                        | [![Octocat](./figures/github.svg)](https://github.com/tianzhaotju/CODA)                         |                                                                  |
| 2024 | JSEP            | CodeBERT‐Attack: Adversarial attack against source code deep learning models via pre‐trained model.                |                                                                                                 |                                                                  | 


### Other Threats
This includes xxx

## 🛡Defensive Strategies
In response to the growing security threats, researchers have proposed various defense mechanisms:

### Backdoor Defense
Methods for defending against backdoor attacks include:

| Year | Conf./Jour. | Paper                                                                                   | Code Reporisty                                                                      | Reproduced Reporisty |  
|------|-------------|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|----------------------|
| 2022 | ICPR        | Backdoors in Neural Models of Source Code.                                              | [![Octocat](./figures/github.svg)](https://github.com/goutham7r/backdoors-for-code) |                      |
| 2023 | CoRR        | Occlusion-based Detection of Trojan-triggering Inputs in Large Language Models of Code. |                                                                                     |                      |
| 2024 | TOSEM       | Poison attack and poison detection on deep source code processing models.               |                                                                                     |                      |
| 2024 | CoRR        | Eliminating Backdoors in Neural Code Models via Trigger Inversion.  🚩                  |                                                                                     |                      |



### Adversarial Defense
Approaches to counter adversarial attacks include:

| Year | Conf./Jour. | Paper                                                                                         | Code Reporisty                                                            | Reproduced Reporisty |    
|------|-------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------|
| 2021 | SANER       | Semantic Robustness of Models of Source Code.                                                 | [![Octocat](./figures/github.svg)](https://github.com/jjhenkel/averloc)   |                      |
| 2022 | COLING      | Semantic-Preserving Adversarial Code Comprehension.                                           | [![Octocat](./figures/github.svg)](https://github.com/EricLee8/SPACE)     |                      |
| 2023 | ICSE        | RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation. | [![Octocat](./figures/github.svg)](https://github.com/RoPGen/RoPGen)      |                      |
| 2023 | PACM PL     | Discrete Adversarial Attack to Models of Code.                                                | [![Octocat](./figures/github.svg)](https://github.com/)                   |                      |
| 2023 | CCS         | Large language models for code: Security hardening and adversarial testing.                   | [![Octocat](./figures/github.svg)](https://github.com/eth-sri/sven)       |                      |
| 2023 | CoRR        | Enhancing Robustness of AI Offensive Code Generators via Data Augmentation.                   | [![Octocat](./figures/github.svg)](https://github.com/)                   |                      |


[//]: # (## Challenges and Future Directions)

[//]: # (The research points to significant **challenges** in securing NCMs, including the difficulty of detecting sophisticated attacks and the lack of standardized **benchmarks** for evaluating defenses. Future work should focus on:)

[//]: # (- **Creating robust defense frameworks** for both **adversarial** and **backdoor** attacks.)

[//]: # (- Developing **evaluation standards** for testing the security of NCMs.)

[//]: # (## ⚙ Datasets and Open-Source Tools)

[//]: # (A set of tools and benchmarks are available to researchers and developers for experimenting with different **security attacks** and **defense strategies** for NCMs.)

## Citation
If you find this repository useful for your work, please include the following citation:
```
@article{xxx,
  title={Security of Neural Code Models},
  author={xxx and xxx and xxx and xxx and xxx},
  journal={arXiv preprint arXiv:xxxxx},
  year={2024}
}
```
