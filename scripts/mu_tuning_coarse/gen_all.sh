#!/bin/bash

homedir=$(pwd)
for dir in *_tp*_nflux*/n*/beta*_U*/mu*/
do
    cd ${dir}
    pwd
    chmod +x gen_sim_files.sh && ./gen_sim_files.sh
    cd ${homedir}
done