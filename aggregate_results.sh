#!/bin/bash
echo "config,method,TA,RA,forget_acc,UA"
for log in logs/exp_FT_*_run1.log logs/exp_GA_*_run1.log logs/exp_MUNBa_*_run1.log logs/exp_RL_*_run1.log logs/exp_SalUn_*_run1.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log | sed 's/^exp_//')
    method=$(echo "$name" | cut -d_ -f1)
    cfg=$(echo "$name" | sed 's/^[^_]*_//; s/_run1$//')
    ta=$(grep "TA (Test" "$log" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    ra=$(grep "RA (Remaining" "$log" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    fa=$(grep "^forget acc:" "$log" | tail -1 | awk '{print $NF}')
    ua=$(awk -v fa="$fa" 'BEGIN {if(fa!="") printf "%.2f", 100-fa; else print ""}')
    echo "${cfg},${method},${ta},${ra},${fa},${ua}"
done
