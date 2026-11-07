#!/bin/bash
# Reproduce the whole v1 (final) model from scratch: results are saved to RESULTS_DIR (default: results)
set -e
python3 01_stage1_select.py         > "${RESULTS_DIR:-results}/log_stage1.txt"   2>&1
python3 01b_stage1_alternatives.py  > "${RESULTS_DIR:-results}/log_stage1b.txt"  2>&1
python3 02_stage2_risk.py           > "${RESULTS_DIR:-results}/log_stage2.txt"   2>&1
python3 03_select_lambda.py         > "${RESULTS_DIR:-results}/log_stage3.txt"   2>&1
python3 04_final_test.py            > "${RESULTS_DIR:-results}/log_stage4_test.txt" 2>&1
echo "done"
