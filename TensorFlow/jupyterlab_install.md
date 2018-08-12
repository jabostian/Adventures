### Jupyterlab Configurations
In order to run applications for each TensorFlow configuration through the
Jupyter Notebook / Jupyterlab UI, you have to install Jupyterlab to both the
**tf_cpu** and **tf_gpu** conda environments.  Jupyter has a lot of dependencies,
so there will be a lot of packages installed to each environment.

I prefer to use Jupyterlab over vanilla Jupyter, so install like this:
```
conda install -c conda-forge jupyterlab
```

If you want to have Jupyterlab running at the same time in each environment,
you'll have to specify a unique port number for each instance, and make sure to
use the right URL from the browser for the TensorFlow version you want to run.

The easiest way to manage all of this is to create a Jupyter config file for
each environment, and set the necessary properties there.  Start by creating
a configuration like this:
```
jupyter lab --generate-config
```

This will create a configuration in ```$HOME/.jupyter/jupyter_notebook_config.py```.
Since we want two versions of this, rename the config file to align with one of
the TensorFlow conda environments.  Start with the GPU version:
```tutorial
cd $HOME/.jupyter
mv jupyter_notebook_config.py gpu_config.py
```

Now edit ```gpu_config.py```, and change these settings:
```
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
```

The first line tells Jupyterlab which port to listen to.  **8888** is the default
port, and you should set it to another number if you know that this will conflict
with another server on your system.

The second line disables the default behavior of opening up a web browser when the
server starts.  This isn't strictly necessary, but I like to open the web browser
myself with the port number of the environment I want just to be sure I know which
TensorFlow instance I'm talking to.

Now copy ```gpu_config.py``` to ```cpu_config.py```, and change the port number
like this:
```
c.NotebookApp.port = 8889
```

Again, you can make the port number whatever you want, as long as it's not the
same as the port number for the other Jupyterlab configuration.

Now, start the jupyterlab server with the proper configuration from the
conda environment you want to use.  For example, if you're working with the GPU
version (tf_gpu), start the server like this:
```
jupyter lab --config=~/.jupyter/gpu_config.py
```

In a similar way you can start the CPU version (tf_cpu) like this:
```
jupyter lab --config=~/.jupyter/cpu_config.py
```

Use the URL that the server puts out at the console, and you can now run notebooks
for each TensorFlow instance side-by-side.

_**Just be careful that if you run the same notebook from each configuration
that only one kernel is active for that notebook at a time.  Otherwise,
un-predictable results will occur.**_
