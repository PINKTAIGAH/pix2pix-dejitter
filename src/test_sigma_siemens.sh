sigma_array=(30 25 20 15 10 5 2)

for i in ${sigma_array[@]}; do 
    echo "# Adding $i as SIGMA" >> config.py
    echo "SIGMA = $i" >> config.py
    echo "EVALUATION_IMAGE_FILE= '../evaluation/siemens_sigma_$i'" >> config.py
    mkdir ../evaluation/siemens_sigma_$i
    python3 train.py 
    python3 evaluation_siemens.py
    cp ../models/gen.pth.tar ../models/gen.siemens_sigma_$i.tar
    cp ../models/disc.pth.tar ../models/disc.siemens_sigma_$i.tar
    rm ../evaluation/default/*
done
