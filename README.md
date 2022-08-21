# Trainlib

## Installation

For editable install with development requirements use `make install`,
to only install the library run `pip install .`

## Concepts
The goal of `trainlib` is to reduce biolerplate code by providing working routines
for training neural networks. Repeating tasks, such as implementation of metrics,
logging and visualizations are all implemented in `trainlib` so one can start focusing
on hyperparamter tuning and model architectures. However, to ensure `trainlib` will
work out-of-the-box it has strong assumptions that must be met.

**Filepaths are in a CSV**  
Filepaths for images and labels are given in a CSV file, with different CSV files
for training, validation and testing. Other formats, such as JSON are not supported.  

**Everything configurable is in a YAML file**  
Every training is configured in a `config.yaml`.
This includes filenames for the CSV files, hyperparamters, definition of model architectures,
loss functions, optimizers, transform pipelines. By strict adherence to his concept, it is
possible to reproduce results from a training with only the config file.

**Custom code is monkey patched**  
`trainlib` allows to monkey patch transforms, model architectures, loss and optimizers by
providing a python script with patched code in the config. It will then try to load
the function form the patch-script first and only fall back to default parameters
if this fails. For transforms, it will try to load every transform given in the config
from the patch-script. For model, loss and optimizer a `get_model`/`get_loss`/`get_optimizer`
function is expected in the patch-script.

**Reproducibility**  
With each new training, `trainlib` will copy **everything** to an output directory.
This includes configurations, the complete codebase and all patches.
This way, the results can be reproduced later on, even if the library did change in the meantime.

## Features
`trainlib` comes with already implemented:
- training routine for a U-Net segmentation model for medical images
- Logging of Dice coefficient, Hausdorff distance, surface distance and loss
- Logging of intermediate stages for segmentation
- Creating of simple report after training, including a GIF of segmentation progress
- Logging of CO2 emmission with [codecarbon](https://github.com/mlco2/codecarbon)
- Checkpointing of models after each improvement
- possibility to resume from checkpoints
- Enhanced viewing functions for mecical images inside jupyter notebook
- Send training updates to mobile phone using [pushover](https://pushover.net/)
- enhanced debugging features, designed to help catch commong errors early


## ToDo

 - [x] data.py  
 - [x] transforms.py  
 - [x] trainer   
 - [ ] Support MONAI bundle
 - [ ] enable ensemble training  
 - [ ] Enable hyper parameter tuning with ray
 - [x] argparse to train.py
 - [x] add val loss (tqdm and ~~loss logs~~)
 - [x] add CO2 estimation (https://github.com/mlco2/codecarbon#quickstart-)
 - [ ] add final classification report
 - [ ] Rewrite trainer.py to --> base trainer and subclasses segmentation trainer and classification trainer
 - [ ] add support for 2d segmentation
 - [ ] add support for 3d classification
 - [ ] add support for 2d classification
 - [ ] add project set-up (create folders for data/notebooks/patch, ci.yaml, Makefile etc. by running a single terminal command)
 - [ ] switch from setup.py to .toml 



## Known issues

### `RuntimeError: received 0 items of ancdata`
This error occurs, because the number of allowed open files is to low.
How the max. number of open files can be increased is described in: https://superuser.com/questions/1200539/cannot-increase-open-file-limit-past-4096-ubuntu

> ## In Summary
> If you want to increase the limit shown by ulimit -n, you should:
>     
> ```bash
>     Modify /etc/systemd/user.conf and /etc/systemd/system.conf with the following line (this takes care of graphical login):
>
>     DefaultLimitNOFILE=65535
>
>     Modify /etc/security/limits.conf with the following lines (this takes care of non-GUI login):
>
>     <enter your username> hard nofile 65535
>     <enter your username> soft nofile 65535
> ```
>  
>  
>    Reboot your computer for changes to take effect.

### `RuntimeError: DataLoader worker (pid(s) 6662) exited unexpectedly`
Most likely because of too much RAM while caching transforms.
Fix this error by reducing the batch_size, image size (introduce downsizing early) or number of transforms.

also see this issue:  https://github.com/pytorch/pytorch/issues/13246
