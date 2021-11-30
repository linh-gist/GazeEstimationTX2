cd mtcnn
rm -rf chobj/ create_engines det1.engine det2.engine det3.engine
make
./create_engines
cd ..
rm -rf pytrt.cpython*
make
