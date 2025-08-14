#!/bin/bash
echo curl -X POST \"http://localhost:8009/analyze?input_path=/app/data/$1.csv\&out_dir=/app/out/$1\"
curl -Xv POST "http://localhost:8009/analyze?input_path=/app/data/$1.csv\&out_dir=/app/out/$1"
