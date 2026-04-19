#!/bin/bash
homedir=$(pwd)
for dir in *_nflux*/n*/
do
    cd ${dir}
    chmod 775 get_mu.sh
    ./get_mu.sh
    cp best* ${homedir}/
    cd ${homedir}
done