
for l in `seq 1 10`; do
for t in 1 2 3; do

python ../generator/gen2.py $t 10475 25 $l 100 10 | tr -d '\[' | tr -d '\]' > temp.txt
cat temp.txt | sed -n "1,500p" > selectiondata16/train-$t-$l.txt
cat temp.txt | sed -n "501,10500p" > selectiondata16/test-$t-$l.txt


lval=99999999999
minf=-1

for net in 3; do
for rep in `seq 1 3`; do


python main.py mine mine_net${net} ../log/mnist_test selectiondata16/train-$t-$l.txt --objective one-class --lr 0.00001 --n_epochs 10000 --lr_milestone 9900 --batch_size 500 --weight_decay 1e-8 --pretrain False --normal_class 0 > temp-${rep}.txt

loss=`(cat temp-${rep}.txt | grep "LOSS" | cut -d' ' -f2)`
if [ 1 -eq "$(echo "${loss} < ${lval}" | bc)" ]; then
    lval=$loss
    minf=$rep
fi
echo "$minf, $lval"


done

cat temp-${minf}.txt | grep -v "LOSS" > results16/nn-$t-$l-$net.txt


done
done
done

