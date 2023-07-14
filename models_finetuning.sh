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
        npi_plural
        npi_length
        sva_base_agreement
        sva_base_lexical
        sva_base_plural
        sva_hard_agreement
        sva_hard_lexical
        sva_hard_length
        sva_hard_plural)
# rates=(0.0 0.001 0.01 0.05 0.1 0.2 0.5)

# rates1=(0.001  0.05 0.2 )
rates2=(0.0    0.01 0.1 0.5)

models=(gpt2 lstm-toy lstm-glove t5-base bert-base-uncased roberta-base)
# iterate over rate and run the pipeline

for model in "${models[@]}"
do
    for prop in "${props[@]}"
    do 
        for rate in "${rates2[@]}"
        do 
            echo "------ FINETUNING model $model prop $prop WITH RATE $rate ------"
            python main.py --prop $prop --task finetune --model $model --rate $rate
        done
    done
done