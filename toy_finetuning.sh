
# source ./predicting-venv/bin/activate

# toy 1 task end to end pipeline  
# list of rates 
toy_tasks=(toy_1 toy_2 toy_3 toy_4 toy_5)
rates=(0.0 0.001 0.01 0.05 0.1 0.2 0.5)

# iterate over rate and run the pipeline

for toy in "${toy_tasks[@]}"
do 
	for rate in "${rates[@]}"
	do 
		echo "------ FINETUNING task $toy WITH RATE $rate ------"
		python main.py --prop $toy --task finetune --model lstm-toy --rate $rate
	done
done



