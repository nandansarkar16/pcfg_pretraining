This repo uses the repositories [litgpt](https://github.com/Lightning-AI/litgpt) and [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG) with a few necessary modifications and additional files.  

📄 [Read the project writeup! (PDF)](EnhancingPretrainingDataEfficiencyUsingPCFGs_writeup.pdf)

---

## Setup

```bash
git clone https://github.com/nandansarkar16/pcfg_pretraining.git
cd pcfg_pretraining
```

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

**Pretraining and fine-tuning language models**

Example scripts for pretraining and fine-tuning can be found in:
```bash
./litgpt/scripts/train_50M
```

**BLiMP evaluation after training**

BLiMP evaluation is run automatically at the end of each epoch during training.  
For post-training evaluation, run:
```bash
bash ./litgpt/calc_blimp_after_train.sh
```
Be sure to update the model paths in:
```bash
./litgpt/calc_blimp_after_train_test.py
```
