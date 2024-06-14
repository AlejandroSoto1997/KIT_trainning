#!/bin/bash

_c=0
for l in 0  
do
for t in {16..92..2}
do
for w in 0 
do
    FOLD="L_"$l"/SALT_0.1/TEMP_"$t"/RUN_"$w"/"
    mkdir -p $FOLD/
    if [ $t == 14 ]
    then
        python3 multiply.py "init_configs_1microM/L"$l".dat" "init_configs_1microM/L"$l".top" 895. 40
        cp restart.dat $FOLD/
    else
        t0=$((t-2))
        FOLD0="L_"$l"/SALT_0.1/TEMP_"$t0"/RUN_"$w"/"
        cp $FOLD0/last_conf_MD.dat $FOLD/restart.dat
    fi 

    cp topol.top $FOLD/
    cp input_MD.dat $FOLD/

    cd $FOLD
    sed -i 's/TMP/'$t'/g' input_MD.dat
    oxDNA input_MD.dat > log  
    cd -
    
    _c=$((_c+1))
    if [ $_c == 2 ]
    then
       wait; _c=0
    fi

done
done
done
