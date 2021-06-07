for set in weekends; do


lval=99999999999
minf=-1

for rep in `seq 1 3`; do

nr=`(wc -l ../data/graph-timesdata/train-${set}.csv | cut -d' ' -f1 | awk '{print int($1)'})`

python main.py mine mine_sp ../log/mnist_test ../data/graph-timesdata/train-${set}.csv --objective one-class --lr 0.000005 --n_epochs 2000 --lr_milestone 1800 --batch_size ${nr} --weight_decay 0.5e-8 --pretrain False --normal_class 0 > temp-${rep}.txt

loss=`(cat temp-${rep}.txt | grep "LOSS" | cut -d' ' -f2)`

if [ 1 -eq "$(echo "${loss} < ${lval}" | bc)" ]; then
    lval=$loss
    minf=$rep
fi
echo "$minf, $lval"

done

cat temp-${minf}.txt | grep -v "LOSS" > spresults/nn-$set.txt

done