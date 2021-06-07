for set in weekends; do

python2.7 kernel.py ../data/graph-timesdata/train-${set}.csv 0.95 > spresults/kern-${set}.txt

done