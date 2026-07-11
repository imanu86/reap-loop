#!/bin/bash
source /root/pivotal/arm_common.sh
cd /root/pivotal
export PIVOTAL_P2_ENV="{}"
python3 scripts/run_pivotal_arm.py $ARGS --runs 2 --outdir /root/pivotal/arm1_static > /root/pivotal/arm1.out 2>&1
echo "ARM1_RC=$?" | tee /root/pivotal/arm1.done
