#!/bin/bash


python nar_no_prior_model.py --sections 0 &> nar_no_out0.log  &&
python nar_no_prior_model.py --sections 1 &> nar_no_out1.log  &&
python nar_no_prior_model.py --sections 2 &> nar_no_out2.log  &&
python nar_no_prior_model.py --sections 3 &> nar_no_out3.log 
