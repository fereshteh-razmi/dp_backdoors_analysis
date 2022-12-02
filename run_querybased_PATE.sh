#!/bin/sh

for n_query in {2,40,150,410,850,1450,2900,4700,10000}
do
    for c in {1,.10}
    do
        echo "======================================================="
        echo "--n_query $n_query"
        echo "======================================================="

        python PATE_Student.py --n_query $n_query
    done
done

