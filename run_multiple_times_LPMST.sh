#!/bin/sh

for eps in {0.1,0.5,1.0,2.0,3.0,4.0,6.0,8.0,15,50,1000}
do
    echo "======================================================="
    echo "--eps $eps"
    echo "======================================================="

    for c in {1..10}
    do
        seed=$(( 435*c + 1 ))
        python LPMST.py --rrp_seed $seed --eps $eps
    done
done

