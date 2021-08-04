#!/bin/bash

echo "Subtracting baseline from WISE data"
python3 subtract_host_from_wise.py -s > /dev/null 2>&1
echo "Subtracting synthetic host model from Swift data"
python3 subtract_host_from_swift.py -s > /dev/null 2>&1
echo "Subtracting synthetic host model from P200 data"
python3 subtract_host_from_p200.py -s > /dev/null 2>&1
echo "building lightcurve" 
python3 lightcurve.py -s > /dev/null 2>&1
echo "creating the moneyplot"
python3 moneyplot.py -s > /dev/null 2>&1