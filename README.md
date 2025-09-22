# LLM-KD

## Get Started
```
conda create -n kd python=3.10
conda activate kd

pip install -r requirement.txt

cd train
bash train.sh ./ckpts/llama-3.2-1B-Instruct/ ./logs/llama-3.2-1B-Instruct 1
```