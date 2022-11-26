# ml-classifier-a4

ml-classifier-a4 - machine learning demo


A4_codes.py includes the entire implementation of required functions.
Recommended to use Google Colab or Anaconda environment.
(Note: If you want to try out different deep learning models, Colab offers free GPU
access that may speed up your method (Runtime → Change runtime type))

## The Task

The goal of is to mimic a real-world machine learning scenario
where you are given a limited dataset to learn a classifier so that it can generalize
to unseen test data.
The dataset consists of hand-written digit images taken from the MNIST dataset,
and the task is to retrieve the digit label at a specific location. Fig. 1 provides several
example images. Each image has three sub-images: the top image serves as a pointer
to the middle or bottom images. If the top image is a digit in {0, 1, 2, 3, 4}, then
the label of the whole image is the digit in the middle image (e.g., the first example
in Fig. 1 with the label “0”), otherwise, the top image is a digit in {5, 6, 7, 8, 9}
and the label of the whole image is the digit in the bottom image (e.g., the second
example in Fig. 1 with the label “8”).


In the zip file, we provided a training dataset (A4train.csv) and a validation
dataset (A4val.csv). The first column of each CSV file includes the labels y and
the rest of the columns are features X. They will be similar to the new training and
test datasets used for evaluating your method (see Question 3 for more detail). You
should use the training dataset for training and the validation dataset to check the
generalization capability of your model (as this will be how we call your functions).
Be creative :)

### Question 1

Implement a Python function:

    model = learn(X, y)

that takes an `n × d` input matrix X and an `n × 1` label `vector y` (where each entry is
an integer between 0 and 9, inclusive, representing the image labels), and returns
your model of any type, as long as it can be used to classify new data (see next).

Implement a Python function:

    yhat = classify(Xtest, model)

that takes an `m × d` input matrix Xtest and a model learned by your algorithm,
and returns an `m × 1` prediction `vector yhat`.

The matrices and vectors are represented as NumPy arrays.

The functions must be able to handle arbitrary n > 0 and m > 0. Note that in
this assignment, d = 28 × 28 × 3 = 2352 for a vectorized image.


### Setup

1. Install Anaconda for the python version you will be using, in this case 3.9

    https://www.anaconda.com/products/distribution

2. Setting up the env run:

    conda install -y --file requirements.txt

3. Extract a4data.zip

4. Setup PyCharm (optional)

    cat ~/.conda/environments.txt  # Get the correct env i.e.
    /Users/<user>/opt/anaconda3/envs/.venv_conda

Use the env to setup the python interpreter in preferences, use existing conda (find ./bin/python)
i.e.
/Users/<user>>/opt/anaconda3/bin/python

5. Run python a4code.py
