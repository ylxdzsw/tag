wget https://gasnet.lbl.gov/EX/GASNet-2021.9.0.tar.gz
tar zxvf GASNet-2021.9.0.tar.gz
cd GASNet-2021.9.0

./configure --prefix="$PWD"/gasnet --enable-udp --enable-mpi --enable-pthreads --enable-segment-fast --with-segment-mmap-max=4GB --enable-par --enable-mpi-compat
make

cd ..
git clone --recursive https://github.com/flexflow/FlexFlow.git

cd FlexFlow
mkdir build
cd build
../config/config.linux
make
/home/net/shiwei/.conda/envs/shiwei/lib/
