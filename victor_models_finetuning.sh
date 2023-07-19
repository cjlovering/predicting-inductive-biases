source ./predicting-venv/bin/activate
props=(gap-base-length
        gap-base-plural
        gap-hard-length
        gap-hard-none
        gap-hard-tense
        gap-base-lexical
        gap-base-tense
        gap-hard-lexical
        gap-hard-plural
        npi_lexical
        npi_tense
        npi_plural)
       

rates=(0.0 0.01 0.1 0.5)

models=(gpt2)
# iterate over rate and run the pipeline

for model in "${models[@]}"
do
    for prop in "${props[@]}"
    do 
        for rate in "${rates[@]}"
        do 
            echo "------ FINETUNING model $model prop $prop WITH RATE $rate ------"
            python3 main.py --prop $prop --task finetune --model $model --rate $rate
        done
    done
done