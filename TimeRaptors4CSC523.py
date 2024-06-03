# * /************************************************************/
# * Filename:           TimeRaptors4CSC523.py
# * Purpose:            Time series data analysis.
# * /************************************************************/

# TimeRaptors4CSC523.py, starting code for CSC523 Assignment 4 data prep and
# analysis, new project using a small subset of Hawk Mountain 2017 & 2018
# raptor count data to determine percentage increase or decrease as a
# function of Celsius temperature increase or decrease over the last 24 & 48
# hours. This assignment is a regression assignment like assignment 3.

from arfflib_3_1 import readARFF, writeARFF, projectARFF, joinARFF, ARFFtoCSV
from arfflib_3_1 import deriveARFF, discretizeARFF, wekaCorrelationCoefficent
from arfflib_3_1 import sortARFF, imputeARFF, remapAttributes, Normalize
import sys
import os
import csv
import re
import copy
import subprocess
from datetime import datetime, timedelta # sort instances by observation time.
import matplotlib.pyplot as plt # Added for scatter plot 11/2/2020
from bisect import bisect_left

# THESE IMPORTS NEEDED TO USE SCIKIT-LEARN
# WE ARE JUST USING THE NOMINAL-CLASS REGRESSORS, WHERE THE TARGET
# NOMINAL-CLASS IS THE ONLY NON-NUMERIC CLASS, IN THIS EXAMPLE.
# OTHERS ARE COMMENTED OUT FOR NOW
# SEE https://scikit-learn.org/stable/modules/classes.html
import numpy    # numpy.ravel to flatten the Class attribute value lists.
# https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html
from sklearn.dummy import DummyRegressor   # User for ZeroR-like baselines
from sklearn import tree     # for tree.plot_tree, tree.export_graphviz
import graphviz              # for tree plotting via graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,    \
    BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle  # randomize instance order
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr # Continuous-valued correlation coefficient.
from scipy.stats import PearsonRConstantInputWarning
# ERROR METRICS
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
# https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.pearsonr.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.PearsonRConstantInputWarning.html

# POPULATE REGRESSORLIST per this list of sklearn models,
# See CSC523Example2.py lines 230 to 240 for an in-line example.
# USE random_state=42 FOR EVERY MODEL THAT HAS A random_state PARAMETER!!!
# That is important for consistency with the 'make test' reference files.
# Also, any random seed= or random_state= parameter must be 42 for consistency.
# REFERENCES:
# SOME BASELINE REGRESSORS
# https://scikit-learn.org/stable/auto_examples/classification/plot_regressor_comparison.html
# https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

# LINEAR REGRESSION
# https://scikit-learn.org/stable/modules/linear_model.html#
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# regression.coef_[0], etc for the non-target attributes

# ENSEMBLE regressors
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
#

# BAYES STATISTICAl REGRESSOR (conditional probabilities)
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

# NEAREST NEIGHBOR
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

# REGRESSORLIST is complete as handed out.
# egrep 'arff REGRESSOR' TimeRaptors4CSC523.out.ref | cut -d" " -f4
# That is a single space character in -d" ".
# ENTERING:
#   make REGRESSORS
# also prints out this list. It is so big to exercise the nearest
# neighbor best value from the CSC558 assignment 3 Q5, and also
# to compare speeds for brute, kd_tree and ball_tree.
REGRESSORLIST = [
    "DecisionTreeRegressor(criterion='mse',random_state=42)",
    "DecisionTreeRegressor(criterion='friedman_mse',random_state=42)",
    "DecisionTreeRegressor(criterion='mae',random_state=42)",
    "LinearRegression(normalize=False,copy_X=True)",
    "RandomForestRegressor(random_state=42)",
    "BaggingRegressor(random_state=42)",
    "AdaBoostRegressor(random_state=42)",
    "GaussianProcessRegressor(random_state=42)",
] + ["KNeighborsRegressor(n_neighbors=" + str(n) + ",p=" + str(p)
    + ",algorithm='ball_tree',weights='"+W+"')"
        # n=1098 gives max CC of 0.559063 for TOTAL48
        for n in range(1090,1101) for p in range(2,3)
            for W in ['distance']]  # skipping uniform

class printResultsBest(object):
    '''
    Save the best overall and the best LinearRegression for scatter plot.
    printResultsBest added 11/2/2020 for plotting best cases.
    '''
    def __init__(self, nameOfTrainingFile, nameOfRegressor,
            correlationCoefficient, targetAttributeName,
            testDataAttrs, testdata, targetcorrectdata, targetpredicteddata):
        self.nameOfTrainingFile = nameOfTrainingFile
        self.nameOfRegressor = nameOfRegressor
        self.correlationCoefficient = correlationCoefficient
        self.targetAttributeName = targetAttributeName
        self.testDataAttrs = testDataAttrs
        self.testdata = testdata
        self.targetcorrectdata = targetcorrectdata
        self.targetpredicteddata = targetpredicteddata

__printResultsBestOverall__ = None
__printResultsBestLinearRegression__ = None

__printResultsNumber__ = 0
def printResults(nameOfTrainingFile, nameOfRegressor,
    numberTrainingInstances, numberTestInstances, analyzeTimeDelta,
    correlationCoefficient, meanSquaredError, meanAbsoluteError,
    targetAttributeName, nonTargetNames, regressor,
    testDataAttrs, testdata, targetcorrectdata, targetpredicteddata):
    '''    
    1. nameOfTrainingFile IDs the dataset.
    2. nameOfRegressor example is like 'DummyRegressor(YOURARGS)',
       'DecisionTreeRegressor(YOURARGS', etc., defined in global
       list REGRESSORLIST, which is set above function analyze().
    3&4. numberTrainingInstances and numberTestInstances are self-
       explanatory.
    5. analyzeTimeDelta is found by saving datetime.now()'s return
       value in a local variable start at the start of helpAnalyze(),
       and then passing (datetime.now() - that variable) as this
       argument. It is used only to report search times to sys.stderr.
    6. correlationCoefficient is the arfflib_3_1.'s
       wekaCorrelationCoefficent(tagged tnoign, predicted)
       within helpAnalyze. scipy.stats.pearsonr() is choking
       on constant-valued targets from DummyRegressor.
    7. meanSquaredError is the metrics.mean_squared_error().
    8. meanAbsoluteError is the metrics.mean_absolute_error().
    9. targetAttributeName is tagColumn in this assignment.
    10. nonTargetNames are the non-tnoign labels from helpAnalyze()'s
       nonTargetNames parameters.
    11. regressor is the actual regressor for it has been trained
       (regressor.fit() has completed), used to print regressor structure
       where appropriate.
    12. testdata, targetcorrectdata, targetpredicteddata added 11/2/2020
        for plotting best overall and best LinearRegression results.
    '''
    # printResultsBest added 11/2/2020 for plotting best cases.
    global __printResultsBestOverall__
    global __printResultsBestLinearRegression__
    if (__printResultsBestOverall__ == None
            or correlationCoefficient
            > __printResultsBestOverall__.correlationCoefficient):
        __printResultsBestOverall__ = printResultsBest(nameOfTrainingFile,
            nameOfRegressor, correlationCoefficient, targetAttributeName,
            testDataAttrs, testdata, targetcorrectdata, targetpredicteddata)
    if (isinstance(regressor,LinearRegression) and
            (__printResultsBestLinearRegression__ == None
                or correlationCoefficient
            > __printResultsBestLinearRegression__.correlationCoefficient)):
        __printResultsBestLinearRegression__ = printResultsBest(
            nameOfTrainingFile,
            nameOfRegressor, correlationCoefficient, targetAttributeName,
            testDataAttrs, testdata, targetcorrectdata, targetpredicteddata)
    # print("DEBUG ERROR MS", nameOfRegressor, correlationCoefficient,
        # meanSquaredError, meanAbsoluteError); sys.stdout.flush()
    global __printResultsNumber__
    __printResultsNumber__ += 1
    if '/' in nameOfTrainingFile:
        nameOfTrainingFile = nameOfTrainingFile.split('/')
        nameOfTrainingFile = nameOfTrainingFile[-1].strip() # basename
    border = '*******************************************************'
    print(border)
    # Do the single line print, then labels & confusion matrix.
    # A single line, while long, makes sorting on kappa & egrep better.
    CC = (("%.6f" % correlationCoefficient)
        if (isinstance(correlationCoefficient,float)
            or isinstance(correlationCoefficient,int)) else "None")
    print("DATA" + str(__printResultsNumber__),
        nameOfTrainingFile, "REGRESSOR", nameOfRegressor,
        "TRAIN #", numberTrainingInstances, "TEST #", numberTestInstances,
        "CORR COEF", CC,
        "MSQERROR", "%.6f" % meanSquaredError,
        "MABSERROR", "%.6f" % meanAbsoluteError)
    print(border)
    # The rest is tree plotting that works only for trees.
    # Use "pip install graphviz" on command line if graphviz fails.
    # Also plot formula coefficients for LinearRegression.
    try:
        coef = regressor.coef_
        sys.stdout.write(targetAttributeName + ' = \n')
        for i in range(0, len(coef)):
            sys.stdout.write('    ')
            if i > 0:
                sys.stdout.write('+ ')
            if i < len(nonTargetNames):
                sys.stdout.write(("%.6f" % coef[i]) + ' * '
                    + nonTargetNames[i] + '\n')
            else:
                sys.stdout.write(("%.6f" % coef[i]) + '\n')
    except Exception as nocoef:   # tree plotting not supported, not a tree
        pass
    TreePlotOK = True
    try:
        dot_data = tree.export_graphviz(regressor, out_file=None, 
              feature_names=nonTargetNames,
              class_names=classLabels,
              filled=True, rounded=True,  
              special_characters=True) 
        graph = graphviz.Source(dot_data) 
        gfilename = ''
        for c in nameOfTrainingFile + '_' + nameOfRegressor:
            if c.isalnum() or c == '_':
                gfilename += c
            else:
                gfilename += '_'
        gfileout = os.environ['HOME'] + '/public_html/' + gfilename
        dbgfileout = 'https://acad.kutztown.edu/~'        \
            + os.environ['HOME'].split('/')[-1] + '/' + gfilename \
            + '.pdf'
        graph.render(gfileout)
    except Exception as oopsie:   # tree plotting not supported, not a tree
        TreePlotOK = False
        # if isinstance(regressor, DecisionTreeRegressor):
            # sys.stderr.write("TREE JPG skipped: " + str(oopsie) + '\n')
    if TreePlotOK:
        sys.stderr.write("DEBUG rendered file " +  dbgfileout + '\n')

def preprocess(dataFileName, dataAttributes, dataInstances):
    '''
    dataFileName is the ARFF file read as input,
    dataAttributes is the map of its attributes, and
    dataInstances is its list of instances.
    preprocess preprocesses per STUDENT instructions below,
    returning a revised (attributes, instances) pair.
    '''
    # Parson-supplied error check to ensure datetime is the first column,
    # which is needed for bisect_left to search on a sorted instance list.
    datetime_column = dataAttributes['datetime'][0]
    datetime_type = dataAttributes['datetime'][1]
    if datetime_column != 0:
        # In a more robust solution we could reorder datetime to the front.
        raise ValueError("ERROR, column 0 must be datetime in file "
            + dataFileName)
    if datetime_type != "string":
        # Code below converts to a datetime.datetime object for timedelta.
        raise ValueError("ERROR, column 0 must be datetime TYPE string in file "
            + dataFileName + ", TYPE=" + str(datetime_type))

    # Convert datetime string objs to datetime objs.
    # Iterate over the instances in dataInstances, and for
    # each instance, reassign into instance[0], which is the datetime
    # attribute, the datetime.strptime(...) conversion of instance[0],
    # per "time.strptime(string[, format])" documentation in
    # https://docs.python.org/3/library/time.html#time.strptime
    # and "time.strftime(format[, t])" format string documentation in
    # https://docs.python.org/3/library/time.html#time.strftime.
    # Use datetime.strptime(column0StringValue, FORMATSTRING) to
    # convert strings like this into datetime object, storing
    # the resulting datetime object back into instance[0].
    # Assignment statement within the loop body looks like:
    # instance[0] = datetime.strptime(instance[0], FORMATSTRING)
    # From F20172018_Date_WS_Temp_Vis_Total_Num_Rnd.arff:
    # @data
    # '2018-12-09 14:00:00',3,-1,9,0
    # '2017-09-14 12:00:00',3,17,?,6
    # '2018-10-13 16:00:00',3,10,25,34
    # '2018-11-02 09:00:00',3,16,15,0
    # ETC. Embed these '-', ' ', and ':' chars into the FORMATSTRING
    # at their places.
    for instance in dataInstances:
        instance[0] = datetime.strptime(instance[0], "%Y-%m-%d %H:%M:%S")

    # Sort the dataInstances on their datetime field and
    # store the sorted dataset back into dataInstances. sortARFF returns only
    # the sorted instances, not the attributes, which it doesn't change.
    # def sortARFF(attrmap, dataset, attributeKeys, sreverse=False)
    #   sortARFF returns a sorted copy of dataset, but not the attributes.
    #   attributeKeys can contain attribute names or column indicies
    dataInstances = sortARFF(dataAttributes, dataInstances, ['datetime'])
    
    # Parson-supplied variables, helper functions for the student C loop below.
    # Each of the hours variables measures the distance in time between
    # the instance receiving a lagged 'TOTAL24', 'Temp24', 'TOTAL48', 'Temp48'
    # set of values, and the permitted datetime timestamp on the sending
    # instance. See isValidTime(...) and isValidTemp(...) below.
    # https://docs.python.org/3/library/datetime.html#datetime.timedelta
    hours24 = timedelta(hours=24)
    hours48 = timedelta(hours=48)
    timeMargin = timedelta(hours=4)
    # Every time bisect_left (below) finds a sending instance, record its
    # instance index in last24ix or 24-hour lags or last48ix for 48-hour.
    last24ix = 0
    last48ix = 0
    Temp_column = dataAttributes['Temp'][0]
    TOTAL_column = dataAttributes['TOTAL'][0]
    # Accumulate instance lists with 4 lagged values
    #   TOTAL24, Temp24, TOTAL48, Temp48
    # in newrows using newrows.append(VALUE) and/or newrows.extend(VALUELIST)
    newrows = []
    def isValidTime(dataInstances, instix, priorix, hours24_OR_hours48):
        # hours24_OR_hours48 is hours24 on the 1st call and hours48 on the 2nd
        return (priorix >= 0 and priorix < len(dataInstances)
                and (dataInstances[instix][0] - dataInstances[priorix][0])
                    >= (hours24_OR_hours48 - timeMargin)
                and (dataInstances[instix][0] - dataInstances[priorix][0])
                    <= (hours24_OR_hours48 + timeMargin))
    def isValidTemp(dataInstances, instix, priorix):
        return (dataInstances[instix][Temp_column] != None
            and dataInstances[priorix][Temp_column] != None)

    #       bisect_left searches for what would be the insertion point for
    #       the timestamp [priortime] within dataInstances, limiting the
    #       search to dataInstances[last24ix:instix] for efficiency. See
    #       https://docs.python.org/3/library/bisect.html
    #       https://en.wikipedia.org/wiki/Binary_search_algorithm FIG. 1

    # Make [TOTAL24, Temp24, TOTAL48, Temp48] data.
    # For each instance instix into dataInstances, -- an int index variable {
    #   Assign a reference to that instance into variable 'inst'.
    #   Construct an empty 'newinst' list.
    #   ******************************************************************
    #   Set 'priortime' equal to inst's time MINUS hours24; we will search
    #       back to that approximate time.
    #   Set 'priorix' equal to
    #       bisect_left(dataInstances, [priortime], last24ix, instix)
    #   If isValidTime(dataInstances, instix, priorix, hours24) {
    #       Set last24ix equal to priorix
    #       Set 'delta' = (instix's TOTAL + 1.0)/(priorix's TOTAL + 1.0)
    #           The reason for adding 1.0 is to avoid divide by 0.
    #           We are interested in TOTAL24 ratios that exceed 1.0.
    #       Append delta into newinst
    #       If isValidTemp(dataInstances, instix, priorix) {
    #           Append (instix's Temp MINUS priorix's Temp) into newinst
    #       } Else {
    #           Append 0 into newinst. We are treating unknown Temp deltas as 0.
    #       }
    #   } Else (FAILED the four-way-anded If above) {
    #       Append two 0's into newinst, one for TOTAL24, one for TEMP24.
    #   }
    #   ******************************************************************
    #   REPEAT ABOVE ***-delimited CODE USNG 48 INSTEAD OF 24.
    #   Set 'priortime' equal to inst's time MINUS hours48; we will search
    #       back to that approximate time.
    #   REPEAT THE ABOVE CODE from "Set 'priortime'" thru "Append two 0's",
    #       using 48 instead of 24 hour lags and 44..52 time bounds.
    #   ******************************************************************
    #   newrows.append(newinst) as the final statement within the for loop.
    # }
    for instix in range(0, len(dataInstances)):
        inst = dataInstances[instix]
        newinst = []
        # 24 hours
        priortime = inst[datetime_column] - hours24
        priorix = bisect_left(dataInstances, [priortime], last24ix, instix)
        
        if isValidTime(dataInstances, instix, priorix, hours24):
            last24ix = priorix
            delta = (inst[TOTAL_column] + 1.0) / (
                dataInstances[priorix][TOTAL_column] + 1.0)
            newinst.append(delta)
            if(isValidTemp(dataInstances, instix, priorix)):
               newinst.append(inst[Temp_column] - dataInstances[priorix][Temp_column])
            else:
               newinst.append(0)
        else:
            newinst.append(0)
            newinst.append(0)
                        
        # 48 hours 
        priortime = inst[datetime_column] - hours48
        priorix = bisect_left(dataInstances, [priortime], last48ix, instix)
                
        if isValidTime(dataInstances, instix, priorix, hours48):
            last48ix = priorix
            delta = (inst[TOTAL_column] + 1.0) / (
                dataInstances[priorix][TOTAL_column] + 1.0)
            newinst.append(delta)
            if(isValidTemp(dataInstances, instix, priorix)):
                newinst.append(inst[Temp_column] - dataInstances[priorix][Temp_column])
            else:
                newinst.append(0)
        else:
            newinst.append(0)
            newinst.append(0)
        # append all
        newrows.append(newinst)

    # Call imputeARFF(...) to impute the 'mean' value
    # of that column for attributes WindSpdKmh, Temp, and Visibility.
    # Weka shows 47 ? missing values for WindSpdKmh in the input dataset
    # (Python None in a data field means unknown), 5 missing Temp values,
    # and 244 unknown Visibility values. After experimenting, I decided
    # that 'mean' is the safest approach. 'random' and 'median' create
    # values that are sometimes much too large. imputeARFF returns only
    # the updated instances, not the attributes, which it doesn't change.
    # def imputeARFF(attrmap, dataset, attributeKeys, replacement, seed=None)
    #   imputeARFF returns an imputed copy of dataset, but not the attributes.
    #   attributeKeys can contain attribute names or column indicies
    #   Parameter replacement can be one of 'mean' (works with numerics),
    #   'mode' or 'median' -- works with sortables -- 'min' or 'max'
    #   for sortable attributes, 'random' for a uniform float value between
    #   the min and the max of a numeric attribute, or a function
    #   supplied by the caller. Param seed needed only for 'random'.
    dataInstances = imputeARFF(dataAttributes, dataInstances,
                               ['WindSpdKmh', 'Temp', 'Visibility'], 'mean')
    
    # Use joinARFF(...) to join numeric attributes
    #   TOTAL24, Temp24, TOTAL48, and Temp48 residing in newrows to
    #   four new columns in dataInstances per examples of joinARFF(...)
    #   in your prior assignments.
    #   joinARFF(attrmap, dataset, nameTypePairs, rowsOfNewColumns)
    #   joinARFF returns both (attributes, values) of the new data.
    dataAttributes, dataInstances = joinARFF(dataAttributes, dataInstances,
                                             [('TOTAL24','numeric'),
                                              ('Temp24', 'numeric'),
                                              ('TOTAL48', 'numeric'),
                                              ('Temp48', 'numeric')], newrows)
    
    # Function preprocess(...) ends with the following statement.
    return (dataAttributes, dataInstances)

def analyze(INPUTARFFNAME, INPUTARFFATTRS, INPUTARFFDATA):
    '''
    INPUTARFFNAME is the name of the filtered ARFF file from __main__.
    INPUTARFFATTRS are the attributes of the filtered ARFF file from __main__.
    INPUTARFFDATA are the data instances of the ARFF file from __main__.
    '''
    # Use as nonTargetNames argument to printResults & regression listings.
    # Use targetNames for the three different target attributes.
    # We are keeping a single nontargetDATA dataset in this
    # assignment, but we are correlating it to three different targets,
    # TOTAL, TOTAL24, and TOTAL48. MAKE SURES TO RAVEL THE TARGETS ONLY.
    indexToName = remapAttributes(INPUTARFFATTRS) # column order
    nonTargetNames = []
    targetNames = ['TOTAL', 'TOTAL24', 'TOTAL48'] # plural targetNames
    for k in sorted(indexToName.keys()):
        if not indexToName[k][0] in targetNames:
            nonTargetNames.append(indexToName[k][0]) # attribute column name

    
    # NOTE: Provided code sets nonTargetNames and targetName via
    # analyze's enclosing variables, no need to provide those last 2
    # args in the call to helpAnalyze.
    # I am making parameter names more generic, not tied to Assn3.
    def helpAnalyze(nameOfTrainingFile, nameOfRegressor, regressor,
        traindataNOtags, traindataONLYtags,
        testdataNOtags, testdataONLYtags, nontargetATTRS,
        nonTargetNames, targetName):
        '''
        The workhorse of analyze() that takes a dataset + regressor
        combination, learns & tests the correlations, and calls
        printResults() to output results to sys.stdout (standard output)
        in a predefined order for text diffing.
        nameOfTrainingFile is the name of the training dataset file.
        nameOfRegressor example is like 'DummyRegressor(YOURARGS)',
            'DecisionTreeRegressor(YOURARGS)', etc., defined in global
            list REGRESSORLIST, which is set above function analyze().
        regressor is the actual regressor object that has been .fit().
        traindataNOtags is the training data without tagColumn.
        traindataONLYtags is the raveled training column with only tagColumn.
        testdataNOtags is the test data without tagColumn.
        testdataONLYtags is the raveled test column with only tagColumn.
        nontargetATTRS attributes for traindataNOtags & testdataNOtags.
        nonTargetNames give the names for all attributes in
            the dataset except targetName.
        targetName is the parameter bound to tagColumn target attribute name.
            targetName is singular.
        '''
        # nonTargetNames and current targetName must be passed as arguments.
        startTime = datetime.now()

        # Train (fit()) the regressor to the training data
        # and training tags same as Assignment 3.
        # You should be able to use your code from Assignment 3 in helpAnalyze()
        regressor.fit(traindataNOtags, traindataONLYtags)
               
        # TEST (.predict()) THE TEST DATASET
        # AGAINST THE TRAINED MODEL, see Assignment 3's predict().
        predictClasses = regressor.predict(testdataNOtags)
        
        # Invoke, on separate command lines,
        # mean_squared_error() and mean_absolute_error() from sklearn.metrics,
        # and wekaCorrelationCoefficent(), as imported from arfflib_3_1,
        # each returning its result as a float value.
        # Each takes two arguments, the expected test data target values as
        # passed in an argument to helpAnalyze(), and the predicted
        # target values as returned by predict().
        mse = mean_squared_error(testdataONLYtags, predictClasses)
        mae = mean_absolute_error(testdataONLYtags, predictClasses)
        cc = wekaCorrelationCoefficent(testdataONLYtags, predictClasses)
               
        # Then call printResults() with the correct
        # arguments, to print the results for this data-regressor pair.
        # printResult's analyzeTimeDelta parameter is
        # (datetime.now() - startTime).
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
        # print("DEBUG CC", CC, type(testdataONLYtags), type(predictTargetValues), testdataONLYtags, predictTargetValues) ; sys.stdout.flush()
        '''
        def printResults(nameOfTrainingFile, nameOfRegressor,
        numberTrainingInstances, numberTestInstances, analyzeTimeDelta,
        correlationCoefficient, meanSquaredError, meanAbsoluteError,
        targetAttributeName, nonTargetNames, regressor,
        testDataAttrs, testdata, targetcorrectdata, targetpredicteddata):
        '''
        aTime = datetime.now() - startTime
        printResults(nameOfTrainingFile, nameOfRegressor,
                     len(traindataONLYtags), len(testdataONLYtags), aTime,
                     cc, mse, mae,
                     targetName, nonTargetNames, regressor,
                     None, None, None, None)
        return None     # This is the end of helpAnalyze, leave intact.

    # start of the main code for analyze():

    # shuffle INPUTARFFDATA to randomize dates,
    # random_state=42) Otherwise we would be training on 2017 instances and
    # testing on 2018. We need a better cross section.
    # shuffle returns a shuffled copy of the data instances.
    INPUTARFFDATA = shuffle(INPUTARFFDATA, random_state=42)
    
    nontargetATTRS, nontargetDATA = projectARFF(INPUTARFFATTRS, INPUTARFFDATA,
            targetNames + ['datetime'],False)
    nonTargetNames.remove('datetime')

    # project TOTAL, TOTAL24, and TOTAL48 as separate
    # targets into separate, one-column instances, then apply ravel() to
    # each of them. We are keeping a single nontargetDATA dataset in this
    # assignment, but we are correlating it to three different targets,
    # TOTAL, TOTAL24, and TOTAL48. MAKE SURE TO RAVEL THE TARGETS.
    # See projectARFF(...) followed by ravel(...) in previous assignments.

    TOTALtargetAttrs, TOTALtargetData = projectARFF(INPUTARFFATTRS, INPUTARFFDATA,
                                                   ['TOTAL'], True)
    TOTALtargetData = numpy.ravel(TOTALtargetData)

    TOTAL24targetAttrs, TOTAL24targetData = projectARFF(INPUTARFFATTRS, INPUTARFFDATA,
                                                   ['TOTAL24'], True)
    TOTAL24targetData = numpy.ravel(TOTAL24targetData)

    TOTAL48targetAttrs, TOTAL48targetData = projectARFF(INPUTARFFATTRS, INPUTARFFDATA,
                                                   ['TOTAL48'], True)
    TOTAL48targetData = numpy.ravel(TOTAL48targetData)    

    # Find integer half the length of nontargetDATA
    # and use slicing to split first half of instances into nontargetDATAtrain
    # and second half into nontargetDATAtest as in previous assignments.
    nontargetDATAtrain = nontargetDATA[0:int(len(nontargetDATA)/2)]
    nontargetDATAtest = nontargetDATA[int(len(nontargetDATA)/2):]
    
    # Write outer for loop over 3 target attr variants.
    # Note that we are iterating over distinct target attribute columns,
    # whereas previous assignments iterated over distinct non-target data,
    # e.g., RANDomized, NORMalized, and RANDomized_NORMalized non-target data
    # in Assignment 3.
    # OUTER LOOP: For each raveled-target-data-instances, name-of-file
    #   2-tuple in (targetData, targetName):
    #       (TOTALtargetData, INPUTARFFNAME + '_' + "TOTAL"),
    #       (TOTAL24targetData, INPUTARFFNAME + '_' +  "TOTAL24"),
    #       (TOTAL48targetData, INPUTARFFNAME + '_' +  "TOTAL48")
    for (targetData, targetName, name) in (
        (TOTALtargetData, INPUTARFFNAME + '_' + "TOTAL", "TOTAL"),
        (TOTAL24targetData, INPUTARFFNAME + '_' +  "TOTAL24", "TOTAL24"),
        (TOTAL48targetData, INPUTARFFNAME + '_' +  "TOTAL48", "TOTAL48")):
    
        # Use slicing to split first half of targetData
        # targetDataTrain and second half into targetDataTest as before.
        targetDataTrain = targetData[0:int(len(targetData)/2)]
        targetDataTest = targetData[int(len(targetData)/2):]
        
        # Iterate over regressors in INNER LOOP.
        #   INNER LOOP: For each regressor in global REGRESSORLIST:
        #       Construct the regressor object
        #       Call helpAnalyze with the appropriate arguments.
        #       See INNER LOOP in Assignments 2 and 3.
        for r in REGRESSORLIST:
            regressor = eval(r)
            '''
            def helpAnalyze(nameOfTrainingFile, nameOfRegressor, regressor,
            traindataNOtags, traindataONLYtags,
            testdataNOtags, testdataONLYtags, nontargetATTRS,
            nonTargetNames, targetName):
            '''
            helpAnalyze(targetName, r, regressor,
                        nontargetDATAtrain, targetDataTrain,
                        nontargetDATAtest, targetDataTest, nontargetATTRS,
                        nonTargetNames, name)

    return None     # Done!

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("USAGE: python TimeRaptors4CSC523.py INFILE.arff.arff OUTFILE.arff\n")
        sys.exit(1)
    infilename = sys.argv[1]
    outfilename = sys.argv[2]
    if not infilename.endswith('.arff'):
        sys.stderr.write("ERROR, file " + infilename + " must end in .arff.\n")
        sys.exit(3)
    if not outfilename.endswith('.arff'):
        sys.stderr.write("ERROR, file " + outfilename + " must end in .arff.\n")
        sys.exit(3)

    dataAttributes, dataInstances = readARFF(sys.argv[1])
    dataAttributes, dataInstances = preprocess(infilename,
        dataAttributes, dataInstances)
    writeARFF(outfilename, dataAttributes, dataInstances, clobber=True,
        floatFormat="%.6f") # floatFormat is new, use 6 decimal place floats
    analyze(outfilename, dataAttributes, dataInstances)

    sys.exit(0)
