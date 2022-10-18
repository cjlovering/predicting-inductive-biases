### Predicting Inductive Biases of Pre-Trained Models

```
@inproceedings{
lovering2021predictinginductive,
title={Predicting Inductive Biases of Fine-tuned Models},
author={Charles Lovering and Rohan Jha and Tal Linzen and Ellie Pavlick},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=mNtmhaDkAr}
}
```

### Setup (2022+)

This project was tested (and updated) to run on v100s with pytorch 1.12.1+cu102 and transformers 4.23.1 using slurm. We saved the full details of the environment in requirements_2022.txt.


```bash
# Create new env.
interact -q gpu-he -m 128GB -g 1 -t 12:00:00 -f v100
module load python/3.7.4 cuda/11.7.1 gcc/10.2
python -m venv predicting-venv
source predicting-venv/bin/activate
pip install --upgrade pip
pip install torch torchvision pytest tqdm pandas gputil spacy[cuda102] transformers pytorch_lightning inflect sklearn wandb nltk plac torchmetrics sentencepiece
pip install plac --upgrade
python -m spacy download en_core_web_lg
bash setup.sh
wget https://nlp.stanford.edu/data/glove.6B.zip
mv glove.6B.zip data/glove/glove.6B.zip 
cd data/glove/
unzip glove.6B.300d.zip
cd ../..
```

Set `wandb` subscription key in your `~/.bash_profile`.

```bash
# This is not the real key. (I set mine up in my .bashrc.)
export WANDB_API_KEY=628318530717958647692528
```

Generate experiments & run!

```bash
# generate datasets
./setup.sh
sbatch datasets.sh
pytest test.py

# generate jobs
bash slim_pipeline.sh # smaller set of settings

# full set of jobs (~day+ of gpu compute)
bash pipeline.sh
```


### Setup (Original)

Install requirements.

```bash
# Create new env.
conda create --name features python=3.8
conda activate features

# Install pytorch.
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install further reqs.
pip install torch torchvision pytest  tqdm pandas gputil spacy[cuda102] transformers pytorch_lightning  pyinflect sklearn wandb nltk plac
pip install plac --upgrade
python -m spacy download en_core_web_lg
```

Set `wandb` subscription key in your `~/.bash_profile`.

```bash
# This is not the real key.
export WANDB_API_KEY=628318530717958647692528
```

Generate experiments & run!

```bash
# generate datasets
./setup.sh
# approx <30 min
sbatch datasets.sh
pytest test.py

# generate jobs
python job.py --experiment finetune
python job.py --experiment probing

# run jobs
sbatch jobs/[DATE]/jobs.sh
```

## Troubleshooting

If you have issues with `plac` (e.g. `plac.opt` is not defined) reinstall it with `pip install plac --upgrade`.

If you have issues with `cupy` uninstall (`pip uninstall cupy-cuda102`) and then re-install (`pip install cupy-cuda102`). 

Let us (@cjlovering, @rohjha) know if you have any questions.
