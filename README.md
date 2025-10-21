Install Dependencies
Boost Libs



sudo apt install libboost-dev libboost-system-dev libboost-filesystem-dev libboost-chrono-dev
OpenCL headers and ICD loader



sudo apt install ocl-icd-opencl-dev


sudo apt install opencl-headers
Python tools



pip install --upgrade --force-reinstall numpy pandas scikit-learn scipy setuptools


Simple Build
Configure build folder and cmake, make sure you have cmake 3.8 or above



cmake -DUSE_ROCM=1 -B build -S .     - for Rocm 6.4
cmake -DUSE_ROCM=1 -B build -S . -D CMAKE_PREFIX_PATH=/opt/rocm     - for Rocm 7.0
Compile the lightGBM



make -j
Build and Install python package



export CMAKE_PREFIX_PATH=/opt/rocm
./build-python.sh install --rocm


<img width="1681" height="955" alt="image" src="https://github.com/user-attachments/assets/46f4b9f0-87a4-4f7a-8e16-408ed746520e" />
