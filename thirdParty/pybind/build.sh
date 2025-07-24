mkdir -p build && cd build

python_prefix=$(python -c "import sys; print(sys.prefix)")  
python_include=${python_prefix}/include/python3.9/
python_lib=${python_prefix}/lib/libpython3.9.so
python_exe=${python_prefix}/bin/python
python_env=${python_prefix}/lib/python3.9/site-packages/
numpy_include=$(python -c "import numpy; print(numpy.get_include())")  

install_path=$(pwd)/../../install

echo $install_path

cmake .. -DPYTHON_INCLUDE_DIRS=${python_include} \
         -DPYTHON_LIBRARIES=${python_lib} \
         -DPYTHON_EXECUTABLE=${python_exe} \
         -DBoost_INCLUDE_DIRS=${install_path}/include/boost \
         -DBoost_LIBRARIES=${install_path}/lib/libboost_python39.so \
         -DORB_SLAM2_INCLUDE_DIR=${install_path}/include/ORB_SLAM2 \
         -DORB_SLAM2_LIBRARIES=${install_path}/lib/libORB_SLAM2.so \
         -DOpenCV_DIR=${install_path}/lib/cmake/opencv4 \
         -DPangolin_DIR=${install_path}/lib/cmake/Pangolin \
         -DPYTHON_NUMPY_INCLUDE_DIR=${numpy_include} \
         -DCMAKE_INSTALL_PREFIX=${python_env}

make install -j