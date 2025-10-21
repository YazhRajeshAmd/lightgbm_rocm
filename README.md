**LIGHTGBM ROCM HOME CREDIT RISK PREDICTION**


This code implements a binary classification model using LightGBM for a Home Credit default risk prediction problem. 

Here's what each section does:

**Data Loading and Preprocessing**
Loads training data from application_train.csv ( Home Credit Default Risk dataset from Kaggle)
Separates features and target: The target variable is TARGET (indicating loan default), and features exclude TARGET and SK_ID_CURR (customer ID)
Handles categorical features: Automatically identifies object-type columns and converts them to categorical data type for LightGBM

**Data Splitting**
Creates train/validation split: Uses 80% for training, 20% for validation with a fixed random seed (42) for reproducibility

**Model Configuration**

**LightGBM parameters:**
objective: 'binary' - Binary classification task
metric: 'auc' - Uses AUC (Area Under Curve) as evaluation metric
boosting_type: 'gbdt' - Gradient Boosting Decision Trees
learning_rate: 0.05 - Conservative learning rate
num_leaves: 31 - Maximum leaves per tree (controls model complexity)

**Training with Early Stopping**
Early stopping: Stops training if validation metric doesn't improve for 50 rounds
Logging: Prints progress every 50 iterations
Training: Runs up to 1000 boosting rounds but may stop early

**Evaluation**
Prediction: Uses the best iteration (from early stopping) to make predictions
Evaluation: Calculates AUC score on validation set and prints the result

**Final goal** is to predict whether a loan applicant will default on their loan based on their application data.

**Install Dependencies**
**Boost Libs**

sudo apt install libboost-dev libboost-system-dev libboost-filesystem-dev libboost-chrono-dev

**OpenCL headers and ICD loader**

sudo apt install ocl-icd-opencl-dev

sudo apt install opencl-headers

**Python tools**

pip install --upgrade --force-reinstall numpy pandas scikit-learn scipy setuptools

**Simple Build**
Configure build folder and cmake, make sure you have cmake 3.8 or above

cmake -DUSE_ROCM=1 -B build -S .     - for Rocm 6.4
cmake -DUSE_ROCM=1 -B build -S . -D CMAKE_PREFIX_PATH=/opt/rocm     - for Rocm 7.0

**Compile the lightGBM**

make -j
**Build and Install python package**

export CMAKE_PREFIX_PATH=/opt/rocm
./build-python.sh install --rocm

**Run the Python Script:**

~/lightgbm_code# python lightgbm_homecredit.py 

**Example Output:**

Training until validation scores don't improve for 50 rounds
[50]    valid_0's auc: 0.747234
[100]   valid_0's auc: 0.755118
[150]   valid_0's auc: 0.757008
[200]   valid_0's auc: 0.757516
[250]   valid_0's auc: 0.757721
Early stopping, best iteration is:
[232]   valid_0's auc: 0.757778
Validation AUC: 0.7578

<img width="1681" height="955" alt="image" src="https://github.com/user-attachments/assets/46f4b9f0-87a4-4f7a-8e16-408ed746520e" />
