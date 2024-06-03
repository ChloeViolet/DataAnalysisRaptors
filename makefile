#       makefile for TimeRaptors4CSC523 assignment 4 project
#       Dr. Dale Parson. Fall 2020

all:		test

TARGET = TimeRaptors4CSC523
include ./makelib
DPOS := $(DPARSON)
PARSONFILES := $(DPOS)/DataMine
PARSONTMP := $(DPOS)/tmp
# FOLLOWING SET floating point comparison tolerance in diffarff.py
# REL_TOL Relative tolerance of .01%, ABS_TOL Absolute tolerance of 10**-6.
REL_TOL = 0.0001
ABS_TOL = 0.000001

build:

test:	clean
	/bin/bash -c "PYTHONPATH=$(PARSONFILES):.:.. time $(PYTHON) TimeRaptors4CSC523.py F20172018_Date_WS_Temp_Vis_Total_Num_Rnd.arff F20172018_Deltas_WS_Temp_Vis_Total_Num.arff > TimeRaptors4CSC523.out.txt"
	egrep -v '@relation' F20172018_Deltas_WS_Temp_Vis_Total_Num.arff | egrep -v '^%' > F20172018_Deltas_WS_Temp_Vis_Total_Num.tmp
	diff F20172018_Deltas_WS_Temp_Vis_Total_Num.tmp F20172018_Deltas_WS_Temp_Vis_Total_Num.arff.ref > F20172018_Deltas_WS_Temp_Vis_Total_Num.arff.dif
	@echo "OUTPUT F20172018_Deltas_WS_Temp_Vis_Total_Num.arff IS OK"
	egrep COEF TimeRaptors4CSC523.out.txt | sort --stable -n -k13 > metrics.txt
	diff TimeRaptors4CSC523.out.txt TimeRaptors4CSC523.out.ref > TimeRaptors4CSC523.out.dif
	@echo "OUTPUT TimeRaptors4CSC523.out.txt IS OK"

clean:	subclean
	/bin/rm -f junk* *.pyc F20172018_Deltas_WS_Temp_Vis_Total_Num.arff TimeRaptors4CSC523.out.txt metrics.txt
	/bin/rm -f *.tmp *.o *.dif *.out *.csv __pycache__/* 

REGRESSORS:
	egrep 'arff REGRESSOR' TimeRaptors4CSC523.out.ref | cut -d" " -f4

STUDENT:
	grep "STUDENT.*%" TimeRaptors4CSC523.py | sed -e 's/^[^#]*# //' |sort
	grep "STUDENT.*%" TimeRaptors4CSC523.py | wc -l
