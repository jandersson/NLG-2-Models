The following instructions assume that you:
  * are running this on a UNIX system such as Mac or Linux
  * have Python 3 installed

============================================
SETUP
============================================
Open up a command prompt whithin the folder where you extracted the code

(Optional: create a virtualenv for the project)

Install the python requirements from requirements.txt:
pip install -r /path/to/requirements.txt

Set the environment variable for the nltk datasets:
source set_environment_variables.sh

Download the necessary corpus and tools for the NLTK package
python -m nltk.downloader brown universal_tagset punkt maxent_treebank_pos_tagger


============================================
RUNNING THE CODE
============================================

There are two files representing the two models implemented
* SemanticModel.py
* InfGrammarMain.py

To run the code open up a terminal window and use python to run them
There are 2 optional parameters

"-s", "--smoothing"
    The smoothing method to be used
    Default: MLEProbDist
    Possible values: {MLEProbDist, SimpleGoodTuringProbDist, LaplaceProbDist, ELEProbDist}
"-g", "--gramCount",
    N in the n-Gram model
    Default: 3
    Possible values: {2, 3, 4}

examples
python SemanticModel.py -s ELEProbDist -g 3 -f 0.6
python SemanticModel.py -smoothing ELEProbDist -gramCount 3
python InfGrammarMain.py -s ELEProbDist -g 3 -f 0.6
python InfGrammarMain.py -smoothing ELEProbDist -gramCount 3
