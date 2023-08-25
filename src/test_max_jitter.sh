max_jitter_array=($(seq 1 0.5 5))

for i in ${max_jitter_array[@]}; do
    echo "# Adding $i as MAX_JITTER" >> config.py
    echo "MAX_JITTER = $i" >> config.py
    echo "EVALUATION_IMAGE_FILE = '../evaluation/max_jitter_$i'" >> config.py
    echo "EVALUATION_METRIC_FILE = '../raw_data/max_jitter.txt'" >> config.py
    mkdir ../evaluation/max_jitter_$i
    python3 train.py
    python3 evaluation.py
done
