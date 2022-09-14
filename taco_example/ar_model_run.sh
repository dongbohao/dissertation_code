#!/bin/bash


python ar_model.py --sections 0 &> ar_out0.log &
python ar_model.py --sections 1 &> ar_out1.log &
python ar_model.py --sections 2 &> ar_out2.log &
python ar_model.py --sections 3 &> ar_out3.log &
