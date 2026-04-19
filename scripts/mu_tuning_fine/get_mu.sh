#!/bin/bash

echo geometry = VALSETGEOMETRY | tee -a best_mu_VALSETNXxVALSETNY_tpVALSETTP_nfluxVALSETNFLUX_nVALSETNT.txt
echo target n = VALSETNT | tee -a best_mu_VALSETNXxVALSETNY_tpVALSETTP_nfluxVALSETNFLUX_nVALSETNT.txt
echo tp = VALSETTP       | tee -a best_mu_VALSETNXxVALSETNY_tpVALSETTP_nfluxVALSETNFLUX_nVALSETNT.txt
echo dtaumax = DTAUMAX   | tee -a best_mu_VALSETNXxVALSETNY_tpVALSETTP_nfluxVALSETNFLUX_nVALSETNT.txt
echo dmu = VALSETDMU     | tee -a best_mu_VALSETNXxVALSETNY_tpVALSETTP_nfluxVALSETNFLUX_nVALSETNT.txt

homedir=$(pwd)
for dir in beta*_U*/
do
    cd ${dir}
    python3 ${DEV}util/get_mu.py VALSETNT mu*/ | tee -a ${homedir}/best_mu_VALSETNXxVALSETNY_tpVALSETTP_nfluxVALSETNFLUX_nVALSETNT.txt
    pwd | tee -a ${homedir}/best_mu_VALSETNXxVALSETNY_tpVALSETTP_nfluxVALSETNFLUX_nVALSETNT.txt
    cd ${homedir}
done

