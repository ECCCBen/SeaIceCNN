Clone TensorFlow environnement used for the CNN model. Anaconda needs to be installed prior to these steps.

From the Anaconda Prompt (command line):
$ cd PathToGitFolder\SeaIceCNN
$ conda env create -f environment.yml
$ conda activate tf

Instead of (base) to the left, you should see (tf) meaning the environment has been cloned and you can start working within the environment.

To leave the environment:
$ conda deactivate tf