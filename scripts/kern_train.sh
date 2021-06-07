for l in `seq 1 10`; do
for t in 1 2 3; do

for q in `(seq 0.4 0.05 0.9)`; do

python2.7 kernel.py selectiondata14/train-$t-$l.txt $q > results14/kern-$t-$l-$q.txt


done
done
done