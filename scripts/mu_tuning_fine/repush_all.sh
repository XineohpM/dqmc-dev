#!/bin/bash

#put every .h5 file back on the stack
homedir=$(pwd)
for dir in *_nflux*/n*/beta*_U*/mu*/
do
    cd ${dir}
    pwd
    python3 /home/users/phoenixm/dqmc-dev/util/push.py ${homedir}/stack *.h5
    cd ${homedir}
done
