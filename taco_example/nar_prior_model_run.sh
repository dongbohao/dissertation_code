#!/bin/bash


python nar_prior_model.py --sections 0 &> nar_prior_out0.log  &&
python nar_prior_model.py --sections 1 &> nar_prior_out1.log  &&
python nar_prior_model.py --sections 2 &> nar_prior_out2.log  &&
python nar_prior_model.py --sections 3 &> nar_prior_out3.log 
