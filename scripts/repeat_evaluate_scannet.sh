#!/bin/bash  
#This is the basic example to print a series of numbers from 1 to 10.  

for num in {1..100}  
do  
	bash scripts/evaluate_scannet.sh
	sleep 60	
done  

echo "Series of numbers from 1 to 100."
