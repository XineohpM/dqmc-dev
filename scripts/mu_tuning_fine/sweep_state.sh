#!/bin/bash

for dir in beta*_U*/
do
    cd ${dir}
    for subdir in mu*/
    do
        cd ${subdir} 
        pwd
        python3 ${DEV}/util/summary.py *.h5 | grep sweep
 	ls *.h5 | wc -l      
	cd ..
    done
    cd ..
done

