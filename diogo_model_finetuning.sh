source ./predicting-venv/bin/activate
props=( npi_length
        sva_base_agreement
        sva_base_lexical
        sva_base_plural
        sva_hard_agreement
        sva_hard_lexical
        sva_hard_length
        sva_hard_plural)

rates=(0.0 0.001 0.01 0.05 0.1 0.2 0.5)


models=(gpt2)
# iterate over rate and run the pipeline

for model in "${models[@]}"
do
    for prop in "${props[@]}"
    do 
        for rate in "${rates[@]}"
        do 
            echo "------ FINETUNING model $model prop $prop WITH RATE $rate ------"
            python main.py --prop $prop --task finetune --model $model --rate $rate
        done
    done
done