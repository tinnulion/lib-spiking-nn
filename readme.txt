Hi, here is some hints how to start demo.

To test the demo will need:
1) Visual Studio 2013
2) CUDA 7
3) Python 3.4
4) Python Tools 2.1 for Visual Studio 2013

To start the test do the following:
1) Open solution "libSpikingNN.sln"
2) Build it
3) Open project "TestSpikeNN"
4) Update constants:
DLL_FILE_PATH = r'<your-path-to>\libSpikingNN.dll'
TEMP_DIR = r'<any-temporary-folder-to-place-images>'
5) Start script, you`ll see console with some progress indications and window with visualization of spiking neurons.

All visualizations will be saved to <any-temporary-folder-to-place-images>. Don`t forget to remove them =)

Some hints about solution structure.
It has two projects: TestSpikeNN and libSpikingNN. The first one has the only script which is 
responsible for showing progress and calling DLL (libSpikingNN). 


