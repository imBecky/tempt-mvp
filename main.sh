#!/bin/bash
epoch=15
for lr in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8; do
    python main.py --lr $lr --epoch $epoch
done