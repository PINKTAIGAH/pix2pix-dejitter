sigma_array=(30 25 20 15 10 5 2)

for i in ${sigma_array[@]}; do 
    echo "# Adding $i as SIGMA" >> config.py
    echo "SIGMA = $i" >> config.py
    echo "EVALUATION_IMAGE_FILE= '../evaluation/siemens_sigma_$i'" >> config.py
    mkdir ../evaluation/sigma_$i
    python3 train.py 
    python3 evaluation_siemens.py
done
