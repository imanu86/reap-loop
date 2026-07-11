#!/bin/bash
source /root/pivotal/arm_common.sh
cd /root/pivotal
export PIVOTAL_P2_ENV="{$PACE_BASE,\"DS4_PACE_REWIND_ARM_K\":\"0.25\",\"DS4_PACE_REWIND_ARM_H\":\"1.0\",\"DS4_PACE_REWIND_FIRE_K\":\"0.5\",\"DS4_PACE_REWIND_FIRE_H\":\"2.0\",\"DS4_PACE_REWIND_EVERY\":\"16\",\"DS4_PACE_REWIND_MAX\":\"6\",\"DS4_PACE_REWIND_BACKOFF\":\"128\"}"
python3 scripts/run_pivotal_arm.py $ARGS --runs 3 --outdir /root/pivotal/arm3_rewind_aggr > /root/pivotal/arm3.out 2>&1
echo "ARM3_RC=$?" | tee /root/pivotal/arm3.done
