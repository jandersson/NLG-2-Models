Install the python requirements from requirements.txt:
pip install -r /path/to/requirements.txt

Set the environment variable for the nltk datasets:
source set_environment_variables.sh

To run the code open up a terminal window and use python to run

There are 3 optional parameters

"-s", "--smoothing"
    The smoothing method to be used
    Default: MLEProbDist
    Possible values: {MLEProbDist, SimpleGoodTuringProbDist, LaplaceProbDist, ELEProbDist}
"-g", "--gramCount",
    N in the n-Gram model
    Default: 3
    Possible values: {2, 3, 4}
"-f", "--trainingSetFractionSize"
    Set the fraction size of the training model
    Default: 0.7
    Possible values: {Float number between 0 and 1}

examples
python SemanticModel.py -s ELEProbDist -g 3 -f 0.6
python SemanticModel.py -smoothing ELEProbDist -gramCount 3 -trainingSetFractionSize 0.6