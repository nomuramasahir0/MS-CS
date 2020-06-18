#!/bin/bash -exu

# trap signal
trap "exit" INT

B=50
EST_TYPES=(unbiased vr naive upper)
RATIO_VALIDATION=0.3
DATASET=ParkinsonSVM
END_POINT=api.py
SEED_END=1000

for EST_TYPE in ${EST_TYPES[@]}; do

    if [[ "${EST_TYPE}" = "naive" ]]; then
        IS_SOURCE_CONCAT_FOR_NATIVE=1
    elif [[ "${EST_TYPE}" = "unbiased" ]] || [[ "${EST_TYPE}" = "vr" ]]; then
        RATIO_DRE=0.3
        IS_SEPARATE_SOURCE_DENS=1
    fi

    for SEED in $(seq 1 100 ${SEED_END}); do
        if [[ "${EST_TYPE}" = "naive" ]]; then
                python ${END_POINT} ${DATASET} --seed=${SEED} \
                --estimator-type=${EST_TYPE} \
                --optimizer=GPUCB \
                --B=${B} \
                --ratio_validation=${RATIO_VALIDATION} \
                --is_source_concat_for_naive=${IS_SOURCE_CONCAT_FOR_NATIVE}
        elif [[ "${EST_TYPE}" = "unbiased" ]] || [[ "${EST_TYPE}" = "vr" ]]; then
                python ${END_POINT} ${DATASET} --seed=${SEED} \
                --estimator-type=${EST_TYPE} \
                --optimizer=GPUCB \
                --B=${B} \
                --ratio_validation=${RATIO_VALIDATION} \
                --ratio_dre=${RATIO_DRE} \
                --is_separate_source_dens=${IS_SEPARATE_SOURCE_DENS}
        elif [[ "${EST_TYPE}" = "upper" ]]; then
                python ${END_POINT} ${DATASET} --seed=${SEED} \
                --estimator-type=${EST_TYPE} \
                --optimizer=GPUCB \
                --B=${B} \
                --ratio_validation=${RATIO_VALIDATION}
        fi
    done

done
