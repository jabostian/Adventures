### Intro
This is my adventure with installing **TensorFlow** with GPU capabilities.  It's a well documented environment, but is still a complex one.  I prefer to use *Anaconda* for configurations like this, so my experience is from within an *Anaconda* environment.

### System Characteristics:
- 3.5 gHZ Intel Core I5
- 64 GB memory
- Nvidia GTX-1070 graphics card
- 500 GB SSD
- Ubuntu Gnome 18.04
- Anaconda and Python 3.6.5
- Packages build-essential, linux-headers, zip, and unzip installed

### Install the NVIDIA Drivers, Libraries, and Samples
At the time I did this, there were no Debian packages created for 18.04, so this install
is from runfiles and archives.  I have basically followed the instructions at
https://www.tensorflow.org/install/install_linux#NVIDIARequirements.

I was able to find others who blazed this trail before me, and found this blog entry
to be helpful:
https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138

- Set up a conda environment (tf_gpu) to work from with just Python in it.  Work from This
  environment so that we can do an A/B comparison between a GPU build and a vanilla TensorFlow
  install with CPU.

#### Install the CUDA Toolkit
This packages was created before Ubuntu 18.04 became available, so the Ubuntu packages
are not yet available.  Install from the runfile instead of a ```.deb``` via ```apt```.
- _**Cuda Toolkit 9.2**_  Found these instructions: https://linoxide.com/linux-how-to/install-cuda-ubuntu/
  - I Already had build-essential installed, so gcc and linux headers are available
  - Get the latest install runfile from https://developer.nvidia.com/cuda-downloads.
    I got ```cuda_9.2.148_396.37_linux.run```
  - ```chmod +x cuda_9.2.148_396.37_linux.run```
  - ```sudo ./cuda_9.2.148_396.37_linux.run --verbose --silent --driver```
  - ```sudo ./cuda_9.2.148_396.37_linux.run --verbose --silent --toolkit --override```
  - ```sudo ./cuda_9.2.148_396.37_linux.run --verbose --silent --samples```
  - ```sudo chown <user_id>:<user_group> -R $HOME/NVIDIA_CUDA-9.2_Samples```
- Make the following environmental updates:
  - Add ```/usr/local/cuda-9.2/bin``` to ```PATH``` in ```~/.bashrc```
  - Add ```/usr/local/cuda-9.2/lib64``` to ```LD_LIBRARY_PATH``` in ```~/.bashrc```

After this installation, these components will be present:
- CUDA Toolkit 9.2
- NVIDIA drivers 390.48
- NVIDIA samples in $HOME

#### Install the NVIDIA Deep Neural Network library (cuDNN)
Get this from the cuDNN download page https://developer.nvidia.com/rdp/cudnn-download.
Choose the appropriate version for CUDA 9.2.  
-  _**cuDNN v7.1.4 library for Linux**_  Follow the directions to install from a
  tar file at https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
   - ```sudo cp cuda/include/cudnn.h /usr/local/cuda-9.2/include```
   - ```sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.2/lib64```
   - ```sudo chmod a+r /usr/local/cuda-9.2/include/cudnn.h /usr/local/cuda-9.2/lib64/libcudnn*```
- Set up ```CUDA_HOME``` in .bashrc:
   - ```export CUDA_HOME=/usr/local/cuda-9.2```

#### Install the NVIDIA CUDA Profile Tools Interface
Note that the documentation says this:
- ```sudo apt-get install cuda-command-line-tools```

but since I've installed from tar files for 18.04, this is the command that works:
- ```sudo apt-get install libcupti-dev```

#### Install the NVIDIA Collective Communications Library (NCCL)
Download the tar file from https://developer.nvidia.com/nccl/nccl-download.  
Choose _**NCCL 2.2.13 O/S agnostic and CUDA 9.2**_
- Install according to the instructions at
  https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#tar
   - ```mkdir $HOME/tmp```
   - ```cd $HOME/tmp```
   - ```mv $HOME/Downloads/nccl_2.2.13-1+cuda9.2_x86_64.txz .```
   - ```tar -xvf nccl_2.2.13-1+cuda9.2_x86_64.txz```
   - ```cd nccl_2.2.13-1+cuda9.2_x86_64```
   - ```sudo cp include/nccl.h /usr/local/cuda-9.2/include```
   - ```sudo cp -R lib /usr/local/cuda-9.2```
   - ```sudo chmod a+r /usr/local/cuda-9.2/include/nccl.h /usr/local/cuda-9.2/lib /usr/local/cuda-9.2/lib/*```

NVIDIA setup is now complete.  

All of the NVIDIA packages that TensorFlow depends on should now be installed.
Log off and back on, and re-enter the ```tf_gpu``` conda environment to get
everything set up properly.  

Try out some of the NVIDIA samples to verify things before moving on with TensorFlow.
The samples were installed to ```$HOME/NVIDIA_CUDA-9.2_Samples```.  Follow the sample build
instructions at https://docs.nvidia.com/cuda/cuda-samples/index.html#building-samples.

- ```cd $HOME/NVIDIA_CUDA-9.2_Samples```
- ```make```

This builds a lot of stuff in the samples directory.  If everything builds and
some of the samples run without trouble, then it's time for TensorFlow.

### Building TensorFlow from Sources with GPU Support
Build all of TensorFlow from scratch with GPU support.  Start at
https://www.tensorflow.org/install/install_sources.  The build itself takes about 35 minutes
with the CPU pegged.  About 4 GB of memory used at the high water mark.

- Get all of the TensorFlow Source
   - ```git clone https://github.com/tensorflow/tensorflow```
- Install bazel to build the code.  Again, we have to build this from source,
  since the packages for 18.04 are not yet ready.  Work from the instructions at
  https://docs.bazel.build/versions/master/install-compile-source.html
   - ```sudo apt install openjdk-8-jdk```
      - This doesn't work.  Apparently we need Oracle Java instead of openJDK.  See
        this StackOverflow thread:
        https://github.com/tensorflow/tensorflow/issues/7497
         - ```sudo apt remove openjdk-8-jdk```
         - Download the compressed tar file for Linux X/86 AMD64
         - ```tar -xvzf tar zxvf jdk1.8.0_181-linux-x64.tar.gz```
         - ```sudo mv jdk1.8.0_181 /usr/local```
         - Add this to ```.bashrc```:
            - ```export JAVA_HOME=/usr/local/jdk1.8.0_181```
            - ```export PATH=$CUDA_HOME/bin:$JAVA_HOME/bin${PATH:+:${PATH}}```
         - Re-build bazel as below, and copy it to ```/usr/local/bin```
         - Try to build TensorFlow again
   - Download the bazel sources zip file from https://github.com/bazelbuild/bazel/releases.
     This should be _**bazel-0.15.2-dist.zip**_.
   - Download the sha256sum _**bazel-0.15.2-dist.zip.sha256**_ and check the
     hash of the sources zip.  These should match:
      - ```sha256sum bazel-0.15.2-dist.zip**_```
      - ```cat bazel-0.15.2-dist.zip.sha256```
   - Make a bazel build directory in $HOME/bazel, and move all of the downloaded
     bazel parts into it.
      - ```mkdir $HOME/bazel```
      - ```cd $HOME/bazel```
      - ```mv ~/Downloads/bazel* .```
      - ```unzip bazel-0.15.2-dist.zip```
      - ```bash ./compile.sh```
      - ```sudo cp output/bazel /usr/local/bin```
- Make sure that all of the necessary packages are installed.  Since we're doing this
  through an Anaconda environment (tf_gpu), add any packages that are not already
  present.  The complete list is:
  - numpy
  - dev
  - pip
  - wheel

  I had to run this command from tf_gpu to complete my environment:
  - ```conda listall numpy```
  - I have a section in my ```.bashrc``` that looks like
  this now:
  ```
  export CUDA_HOME=/usr/local/cuda-9.2
  export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$CUDA_HOME/extras/CUPTI/lib86
  ```
- Configure the TensorFlow build.  There is a script that will ask several questions
  about how you want it built.
  - ```cd $HOME/git/tensorflow```
  - ```./configure```
  - Take all of the defaults, except for the ones that enable nvidia gpu.  Here are
    the relevant settings:
    ```
    Please specify the CUDA SDK version you want to use.
      [Leave empty to default to CUDA 9.0]: 9.2


    Please specify the location where CUDA 9.2 toolkit is installed. Refer to
      README.md for more details.
      [Default is /usr/local/cuda]: /usr/local/cuda-9.2


    Please specify the cuDNN version you want to use.
      [Leave empty to default to cuDNN 7.0]: 7.1


    Please specify the location where cuDNN 7 library is installed. Refer to
      README.md for more details.
      [Default is /usr/local/cuda-9.2]:
    ```

    and further on:
    ```
    Please specify the NCCL version you want to use. If NCCL 2.2 is not installed,
    then you can use version 1.3 that can be fetched automatically but it may have
    worse performance with multiple GPUs. [Default is 2.2]:


    Please specify the location where NCCL 2 library is installed. Refer to
    README.md for more details. [Default is /usr/local/cuda-9.2]:
    ```
- Perform the build
   - ```$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package```

```WARNING: The following configs were expanded more than once: [cuda]. For repeatable flags, repeats are counted twice and may lead to unexpected behavior.```



-----

#### Useful Nvidia (and related) Commands
- Look for an Nvidia PCI-attached card
   - ```lspci | grep -i nvidia```
- Nvidia systems management interface
   - ```nvidia-smi ...```
- Display the Nvidia driver version
   - ```cat /proc/driver/nvidia/version```
- Enhanced system info from uname
   - ```uname -m && cat /etc/*release```
- Show installed signing keys
   - ```sudo apt-key list ```

#### Other NVIDIA stuff to Install   
- Nsight Eclipse IDE:
   - https://developer.nvidia.com/nsight-eclipse-edition
- NVIDIA Visual Profiler
   - https://developer.nvidia.com/nvidia-visual-profiler

-----

### TensorFlow Installation
Everything starts with a TensorFlow install with GPUs.  First do everything in the NVIDIA toolkit
and deep learning sections above.
- https://www.tensorflow.org/install/install_linux

Install TensorFlow from source, because the binary has a CUDA 8.0 dependency, and we are using the latest CUDA 9.1.  If installing from binaries, and trying to use CUDA 9.1, the tensorflow import from Python will fail like this:
- ```ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory```

Rats.

#### Install TensorFlow from Source
Follow the instructions at https://www.tensorflow.org/install/install_sources

- ```git clone https://github.com/tensorflow/tensorflow```
- Use the latest release(1.5) instead of master:
   - ```git checkout r1.5```
- Install Bazel from custom APT repository.  
   https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu
   - Already have openjdk 8 installed ...
   - ```echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list```
   - ```curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -```
   - ```sudo apt install bazel```
      - This drags along IBM Java (really - after listing openjdk as a dependency??)
   - ```sudo apt upgrade bazel```
- Install TensorFlow Python dependencies
   - I use Anaconda for this.  It's an easier way to manage the configuration of prereq-ed
   packaWARNING: The following configs were expanded more than once: [cuda]. For repeatable flags, repeats are counted twice and may lead to unexpected behavior.ges, but it requires running from a named environment.  Still an easier way to manage
   things.
      - ```conda create --name tf python numpy pip wheel```
      - ```source activate tf```
      - ```pip install dev```
- Install TensorFlow GPU pre-reqs
   - This is already done in the NVIDIA sections above
- Configure the installation

#### Configure the Installation
- Activate your Anaconda env.  Always do this when running TensorFlow
   - ```source activate tf```
- Make sure you're in the tensorflow git repo (~/Gondor/git/tensorflow)
- Run the configure script
   - ```./configure```

This is my configuration session:
   ```
Extracting Bazel installation...
You have bazel 0.9.0 installed.
Please specify the location of python. [Default is /home/joshua/anaconda3/envs/tf/bin/python]:
Found possible Python library paths:
  /home/joshua/anaconda3/envs/tf/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/joshua/anaconda3/envs/tf/lib/python3.6/site-packages]
Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]:
jemalloc as malloc support will be enabled for TensorFlow.
Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: N
No Google Cloud Platform support will be enabled for TensorFlow.
Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: N
No Hadoop File System support will be enabled for TensorFlow.
Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: N
No Amazon S3 File System support will be enabled for TensorFlow.
Do you wish to build TensorFlow with XLA JIT support? [y/N]:
No XLA JIT support will be enabled for TensorFlow.
Do you wish to build TensorFlow with GDR support? [y/N]:
No GDR support will be enabled for TensorFlow.
Do you wish to build TensorFlow with VERBS support? [y/N]:
No VERBS support will be enabled for TensorFlow.
Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]:
No OpenCL SYCL support will be enabled for TensorFlow.
Do you wish to build TensorFlow with CUDA support? [y/N]: Y
CUDA support will be enabled for TensorFlow.
Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.1
Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]:
Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.1]
Do you want to use clang as CUDA compiler? [y/N]:
nvcc will be used as CUDA compiler.
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Do you wish to build TensorFlow with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:

Add "--config=mkl" to your bazel command to build with MKL support.
Please note that MKL on MacOS or windows is still not supported.
If you would like to use a local MKL instead of downloading, please set the environment variable "TF_MKL_ROOT" every time before build.
Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.
Configuration finished
```
I chose to support only the features in this environment that I need.  Sometimes the default choice
is not to include them, and sometimes it is, so be careful with your answers.  Things to highlight:
- Default jemalloc support
- No support for any of the distributed file systems:
   - Google Cloud Platform
   - Hadoop
   - Amazon S3
- No support for:
   - XLA JIT
   - GDR
   - VERBS
   - OpenCL SYSCL
- *Be sure to answer* ```Y``` *for CUDA support.*
   - ```9.1``` for CUDA SDK version
      - Default for install location
   - Default 7.0 for cuDNN
      - Default for install location
   - Looked up my card at https://developer.nvidia.com/cuda-gpus, and it has *6.1* compute capability.
   This is the default.
- No clang for CUDA compiler - use nvcc
- Default gcc
- No MPI support
- Default opt flags
- No ```./WORKSPACE``` for Android builds

#### Build and Install the TensorFlow pip Package
Build the package with GPU support.  This took about 35 minutes on my machine, where all 4 CPUs were
pegged.  The instructions say that the build process uses a lot of memory.  I watched my resources
during the build, and it looked like the high-water mark for memory usage was between 4 and 5 GB.
- Build TensorFlow
   - ```bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 2>&1 | tee build.log```
- Build the pip package for TensorFlow
   - ```bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg 2>&1 | tee build_pip.log```
- Install the local wheel file via pip
   - ```pip install /tmp/tensorflow_pkg/tensorflow-1.5.0rc1-cp36-cp36m-linux_x86_64.whl```

Just to check the installation now that everything is built:
```
(tf) joshua@WOPR:~/Gondor/git/tensorflow$ conda list tensorflow
# packages in environment at /home/joshua/anaconda3/envs/tf:
#
tensorflow                1.5.0rc1                  <pip>
tensorflow-tensorboard    0.1.8                     <pip>
```

#### Smoke Test TensorFlow
```
(tf) joshua@WOPR:~$ python
Python 3.6.4 |Anaconda, Inc.| (default, Dec 21 2017, 21:42:08)
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2018-01-14 10:13:15.973132: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:895] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-01-14 10:13:15.973448: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1105] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:02:00.0
totalMemory: 7.92GiB freeMemory: 7.30GiB
2018-01-14 10:13:15.973475: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:02:00.0, compute capability: 6.1)
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>>
```
Woo Hoo

-----
### Experiences During Attempts at Success ...

Use the Anaconda install method for Python 3.6 ...
- ```conda create -n tensorflow```
- ```source activate tensorflow```
- ```pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl```

Tried to validate the TensorFlow installation with a simple Python application like this:
```python
Python 3.6.0 |Anaconda custom (64-bit)| (default, Dec 23 2016, 12:22:00)
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
```
I get this error:
- ```ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory```

Found this on Stackoverflow:
- https://stackoverflow.com/questions/43558707/tensorflow-importerror-libcusolver-so-8-0-cannot-open-shared-object-file-no#43568508

I tried the first solution and keep getting the problem.  I then tried to install the nvidia-cuda-dev package:
- ```sudo apt install nvidia-cuda-dev```
- ```export LD_LIBRARY_PATH=/usr/local/cuda/lib64L$LD_LIBRARY_PATH```

After starting a new terminal to pick up the environmental changes, I still keep getting the error.  Unfortunately it looks like the comment on Stackoverflow about TensorFlow not supporting CUDA 9.0 is for real.  Looks like I need CUDA 8.0  :-|





------------

#### Install the Deep Learning Neural Network library (cuDNN)
   - https://developer.nvidia.com/cudnn
   - Join the developer network and download all of the doc, along with the necessary packages
   - Install from the installation guide PDF
      - No need to install the Nvidia drivers and toolkit, since that is already done
      - Install from debs
   - Verify the installation
      - ```cd $HOME/Gondor/NVIDIA/cdDNN/samples```
      - ```cp -r /usr/src/cudnn_samples_v7/ .```
      - Complete verification ...<br>
        **PROBLEM**:<br>
         ```... error: use of enum ‘cudaDeviceP2PAttr’ without previous declaration```

         Found this thread: https://devtalk.nvidia.com/default/topic/1006726/cudnnv6-mnist-example-compile-errors/

         Followed the advice by changing line 63 of /usr/include/cudnn.h from
         ```#include "driver_types.h"```
         to ```#include <driver_types.h>```

         It worked.  **Woo Hoo!!**.  First line of mnistCUDNN output:<br>
         ```cudnnGetVersion() : 7004 , CUDNN_VERSION from cudnn.h : 7004 (7.0.4)```
