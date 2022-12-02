#!/bin/sh

for c in {0..199}
do
    echo "======================================================="
    echo "--teacher_id $c"
    echo "======================================================="

    python PATE_Teacher.py --teacher_id $c

done

