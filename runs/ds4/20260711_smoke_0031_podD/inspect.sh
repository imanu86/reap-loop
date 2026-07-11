#!/bin/bash
cd /root/smoke_0031
echo "=== header ==="; head -1 a_tokens.csv
echo "=== first 10 gen tokens: pos,token_id  [a | a2 | b | c] ==="
paste -d'|' <(sed -n '2,11p' a_tokens.csv | cut -d, -f1,2) \
            <(sed -n '2,11p' a2_tokens.csv | cut -d, -f1,2) \
            <(sed -n '2,11p' b_tokens.csv | cut -d, -f1,2) \
            <(sed -n '2,11p' c_tokens.csv | cut -d, -f1,2)
echo "=== a_gen first 160 chars ==="; head -c 160 a_gen.txt; echo
echo "=== b_gen first 160 chars ==="; head -c 160 b_gen.txt; echo
echo "=== a2_gen first 160 chars ==="; head -c 160 a2_gen.txt; echo
