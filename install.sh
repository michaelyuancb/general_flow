# # install PyTorch
# # please always double check installation for pytorch and torch-scatter from the official documentation
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..

# point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..

# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../

