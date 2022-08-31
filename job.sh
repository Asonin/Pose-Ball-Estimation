g++ -I../../test_src/ -I../../include/ -MM  ../../test_src/Try.cpp > ../../test_src/Try.dpp.$$; \
sed 's,\(../../test_src/Try\)\.o[ :]*,\1.o ../../test_src/Try.dpp : ,g' < ../../test_src/Try.dpp.$$ > ../../test_src/Try.dpp; \
rm -f ../../test_src/Try.dpp.$$
g++  -c -o Try.o ../../test_src/Try.cpp  -I../../test_src/ -I../../include/ -pipe -g -Wall
g++ -L../lib/  -o ./job Try.o  -I../../test_src/ -I../../include/ -Wl,-rpath=./:./HCNetSDKCom:../lib -lhcnetsdk