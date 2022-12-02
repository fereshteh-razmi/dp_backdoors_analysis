#!/bin/sh

for sigma in {28.284,5.6,2.82,1.41,0.942,0.7071,0.471,0.353,0.19,0.06,0}
do
    for c in {1..10}
    do
        echo "======================================================="
        echo "--sigma $sigma"
        echo "--program $c"
        echo "======================================================="

        python ALIBI.py --sigma $sigma

    done
done

