````markdown
This repo uses the repositories [litgpt](https://github.com/Lightning-AI/litgpt) and [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG) with a few necessary modifications and additional files.  

📄 [Read the project writeup! (PDF)](EnhancingPretrainingDataEfficiencyUsingPCFGs_writeup.pdf)

---

## Setup
```bash
git clone https://github.com/nandansarkar16/pcfg_pretraining.git
cd pcfg_pretraining
````

## Virtual Environments

**LitGPT environment**

```bash
conda create --name litgpt_env --file litgpt_requirements.txt
conda activate litgpt_env
pip install -e litgpt
```

**TN-PCFG environment**

```bash
conda create --name pcfg_env --file pcfg_requirements.txt
conda activate pcfg_env
pip install -e TN-PCFG
```

---

## Usage

**Training Simple-PCFG**

```bash
bash ./TN-PCFG/train_pcfg.sh
```

**Generating data from PCFG**

```bash
bash ./TN-PCFG/generate_data_from_pcfg.sh
```

```
```
