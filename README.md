# ACM-MM2024--EIRAD




<p align="center">
  <img src="https://github.com/Dilemma111/ACM-MM2024--EIRAD/blob/master/pipeline.png" width="550px">
</p>

## 🗂️ EIRAD Dataset

The dataset is in the link below：[https://github.com/Dilemma111/EIRAD-Dataset]


## 🗂️ Environments

**Configure the environment required by llm-attack and Otter Model：**


**[1]Embodied Model:【 Take **Otter Model** as an example 】**

```
1.Compare cuda version returned by nvidia-smi and nvcc --version. They need to match. Or at least, the version get by nvcc --version should be <= the version get by nvidia-smi.

2.Install the pytorch that matches your cuda version. (e.g. cuda 11.7 torch 2.0.0). We have successfully run this code on cuda 11.1 torch 1.10.1 and cuda 11.7 torch 2.0.0. You can refer to PyTorch's documentation, Latest or Previous.

3.You may install via `conda env create -f environment.yml`. Especially to make sure the transformers>=4.28.0, accelerate>=0.18.0.
```

After configuring environment, you can use the 🦩 Flamingo model / 🦦 Otter model as a 🤗 Hugging Face model with only a few lines! One-click and then model configs/weights are downloaded automatically. Please refer to `Huggingface Otter/Flamingo` for details.

**[2]Attack environment configuration**

We need the newest version of FastChat fschat==0.2.23 and please make sure to install this version. The llm-attacks package can be installed by running the following command at the root of this repository:

`<pip install -e .>`.


**[3]Lavis environmenr configuration**

Install from PyPI and for development, you may build from source

 ```
pip install salesforce-lavis 
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```

**[4]Also add the following code to the Otter model source code**

In the modeling_llama.py,please add the following code in LlamaModel:

```
if inputs_embeds is not None:
   input_ids=None
```

## 🗂️ Framework


```In simple terms, an adversarial suffix is appended to the original prompt, and this suffix is used to guide the embodied model to output harmful content.
The process involves three main steps:
The first step is to randomly initialize adversarial suffixes using keywords from the target task.
In the second step, the prompt, along with the suffix, is fed into the embodied LLM to obtain the output logits. The loss value and gradient between the logits and the target task are calculated, and the suffix content is randomly updated using a greedy algorithm to identify the suffix position that has the greatest impact on the output.
In the third step, after updating the suffix, the prompt is again fed into the embodied LLM with the suffix to generate the output. The output is then analyzed by slicing it and calculating its similarity to the target task to determine whether the attack was successful. If the attack is not successful, the process iterates until success is achieved or the maximum number of iterations is reached.
```

<div align=center>
  <img src="https://github.com/Dilemma111/ACM-MM2024--EIRAD/blob/master/framework.png" width="650px">
</div>


## 🗂️ Running

```cd ACM-MM2024--EIRAD\Otter\pipeline\demo

python3 otter-harmful-demo.py

```

## 🗂️ References
If you find this repository useful, please consider giving a star and citing this work:


```@misc{liu2024exploringrobustnessdecisionleveladversarial,
      title={Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models}, 
      author={Shuyuan Liu and Jiawei Chen and Shouwei Ruan and Hang Su and Zhaoxia Yin},
      year={2024},
      eprint={2405.19802},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2405.19802}, 
      doi = {10.1145/3664647.3680616}
}
