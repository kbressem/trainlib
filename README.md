# AI Template
 > Template for MONAI projects
 
 ## Requirements
 pip install -U "monai[all]" pyyaml munch pandas
 
 
 ## ToDo
 
 - [x] data.py  
 - [x] transforms.py  
 - [x] trainer   
 - [ ] pretrained models  
 - [ ] enable ensemble training  
 - [ ] enable multi gpu training (ray vs. monai?)  
 - [x] argparse to train.py
 - [ ] add val loss (tqdm and ~~loss logs~~)
 - [ ] add CO2 estimation (https://github.com/mlco2/codecarbon#quickstart-)
 - [ ] add final classification report



## Known issues

### `RuntimeError: received 0 items of ancdata`
This error occures, because the number of allowed open files is to low. 
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
Most likely because of too much RAM while caching transforms. Fix this error by reducing the batch_size, ROI-size or number of samples in `rand_crop_pos_neg_label`.
 
also see this issue:  https://github.com/pytorch/pytorch/issues/13246
