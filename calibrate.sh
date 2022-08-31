g++ -I../../test_src/ -I../../include/ -MM  ../../test_src/Calibrate.cpp > ../../test_src/Calibrate.dpp.$$; \
sed 's,\(../../test_src/Calibrate\)\.o[ :]*,\1.o ../../test_src/Calibrate.dpp : ,g' < ../../test_src/Calibrate.dpp.$$ > ../../test_src/Calibrate.dpp; \
rm -f ../../test_src/Calibrate.dpp.$$
g++  -c -o Calibrate.o ../../test_src/Calibrate.cpp  -I../../test_src/ -I../../include/ -pipe -g -Wall
g++ -L../lib/  -o ./calibrate Calibrate.o  -I../../test_src/ -I../../include/ -Wl,-rpath=./:./HCNetSDKCom:../lib -lhcnetsdk