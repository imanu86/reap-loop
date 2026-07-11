#!/bin/bash
source /root/pivotal/arm_common.sh
cd /root/pivotal
export PIVOTAL_P2_ENV="{$PACE_BASE}"
python3 scripts/run_pivotal_arm.py $ARGS --runs 3 --outdir /root/pivotal/arm2_rewind_default > /root/pivotal/arm2.out 2>&1
echo "ARM2_RC=$?" | tee /root/pivotal/arm2.done
