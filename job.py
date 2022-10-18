import datetime
import itertools
import json
import os

import plac


@plac.opt(
    "experiment",
    "experiment name",
    choices=[
        "probing",
        "finetune",
        "slim_finetune",
        "slim_probing",
    ],
)
def main(experiment="finetune"):
    if not os.path.exists("./jobs"):
        os.mkdir("./jobs")

    with open(f"./{experiment}.json", "r") as f:
        settings = json.load(f)
    options = list(itertools.product(*settings.values()))

    jobs = []
    options = [o for o in options if not filter_option_out(*o)]
    for idx, option in enumerate(options):
        job_text = template_option(*option)
        job = setup(job_text, idx)
        jobs.append(job)

    jobs_file = template_file(jobs, experiment)
    jobs_name = datetime.datetime.now().strftime(f"{experiment}-%Y-%m-%d")
    # if True:
    #     # hot fix
    #     jobs_name = datetime.datetime.now().strftime(f"direct-{experiment}-%Y-%m-%d")
    print(jobs_name)
    with open(f"./jobs/{jobs_name}.sh", "w") as f:
        f.write(jobs_file)


def template_file(texts, experiment):
    text = "".join(texts)
    out = f"""#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=09:00:00
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -J {experiment}-%j
#SBATCH --constraint v100

# Specify an output file
#SBATCH -o ./out/{experiment}-%j.out
#SBATCH -e ./err/{experiment}-%j.out
#SBATCH -a 0-{len(texts)}

module load python/3.7.4 cuda/11.7.1 gcc/10.2
python -m venv predicting-venv

{text}
"""
    return out


def setup(text, index):
    return f"""
if [ "$SLURM_ARRAY_TASK_ID" -eq {index} ];
then
{text}fi
"""


def template_option(prop, rate, probe, task, model, seed):
    """Generates the template for an a call to train."""

    out = f"""python main.py \
        --prop {prop} \
        --rate {rate} \
        --probe {probe} \
        --task {task} \
        --model {model} \
        --seed {seed}
"""
    return out


def filter_option_out(prop, rate, probe, task, model, seed):
    """Filters some options OUT!"""
    # e.g. lstm_toy and sva
    if "toy" in model and "toy" not in prop:
        return True
    # e.g. bert and toy
    if "toy" not in model and "toy" in prop:
        return True
    return False


if __name__ == "__main__":
    plac.call(main)
