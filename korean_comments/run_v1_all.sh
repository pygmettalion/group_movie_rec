#!/bin/bash
# 1차(최종) 모델 전체를 처음부터 재현: 결과는 RESULTS_DIR (기본 results) 에 저장
set -e
python3 01_stage1_select.py         > "${RESULTS_DIR:-results}/log_stage1.txt"   2>&1
python3 01b_stage1_alternatives.py  > "${RESULTS_DIR:-results}/log_stage1b.txt"  2>&1
python3 02_stage2_risk.py           > "${RESULTS_DIR:-results}/log_stage2.txt"   2>&1
python3 03_select_lambda.py         > "${RESULTS_DIR:-results}/log_stage3.txt"   2>&1
python3 04_final_test.py            > "${RESULTS_DIR:-results}/log_stage4_test.txt" 2>&1
echo "done"
