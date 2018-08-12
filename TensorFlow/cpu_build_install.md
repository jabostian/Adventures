### Intro
This is how I created a **TensorFlow** 1.9 environment for CPU.  I use *Anaconda*
to manage separate CPU and GPU environments so that I can run the same workload
in each, and compare performance.

Since this is a CPU install, I can use the conda package that already exists
on the Anacond Cloud.  The installation is much simpler than the GPU build.

### System Characteristics:
- 3.5 gHZ Intel Core I5
- 64 GB memory
- NVIDIA GTX-1070 graphics card (n/a)
- 500 GB SSD
- Ubuntu Gnome 18.04
- Anaconda and Python 3.6
- Packages build-essential, linux-headers, clang 6.0, zip, and unzip installed
- Oracle java 8 (n/a)

### Create a Conda Environment
So, these are the packages that you need to install to your conda environment,
with the right versions where needed:
  - cython
  - numpy
  - pip
  - python=3.6.6
  - six
  - wheel
- Create the conda environment (tf_cpu)
   - ```conda create --name tf_cpu python=3.6.6 cython numpy pip six wheel```
- The default version of pip that comes with Anaconda is often downlevel.
  Upgrade pip:
  ```
  source activate tf_gpu
  pip install --upgrade pip
  ```

### Install TensorFlow from the Anaconda Cloud
It's almost trivial to install TensorFlow from the Anaconda Cloud.  The
TensorFlow package can be installed from the _***conda-forge**_ channel like
this:

```conda install -c conda-forge tensorflow```

This will drag in several packages to the tf_cpu environment, which is what
conda is meant for.  See https://anaconda.org/conda-forge/tensorflow for
more details.

Check the version of TensorFlow that got installed just to make sure that it
matches the GPU version for comparison
```
(tf_cpu) joshua@WOPR:~/git/Adventures/TensorFlow/code$ conda list tensorflow
# packages in environment at /home/joshua/anaconda3/envs/tf_cpu:
#
# Name                    Version                   Build  Channel
tensorflow                1.9.0                    py36_0    conda-forge
```

You are now ready to run.
