# True Bilingual NMT

This is the official implementation for the paper: [True Bilingual NMT]().

## Machine Properties

The machine used for training models in this paper has the following properties:

- vCPU: 8
- RAM: 52 GB
- GPUs: 4 (Nvidia Testla T4)
- OS: Ubuntu (20.04)
- bootable disk: 100 GB
- Additional disk: 500 GB

## Connect to the Machine

To connect to the machine, you need to follow these steps:

TODO...


## GPU-related Tools

### cuda:

Cuda 11.3 was installed using the following commands:

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/$ cuda-ubuntu2004.pin
$ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/$ cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

### NCCL
To make cuda faster, you need to install [Nvidia Collective Communication Library](https://developer.nvidia.com/nccl) following these steps found
[here](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html):

- Go to: [NVIDIA NCCL home page](https://developer.nvidia.com/nccl), and complete
the short survey.
- Download the deb package of NCCL, the package name should be
`nccl-local-repo-<os_distribution>-<cuda_version>-<architecture>.deb`.
For example, mine is:
    ```bash
    nccl-local-repo-ubuntu2004-2.11.4-cuda11.4_1.0-1_amd64.deb
    ```
- Install it using the following command:
    ```bash
    $ sudo dpkg -i nccl-local-repo-ubuntu2004-2.11.4-cuda11.4_1.0-1_amd64.deb
    ```
- To make sure it was installed successfully, run the following command:
    ```bash
    $ python
    >>> import torch
    >>> torch.cuda.nccl.version()
    (2, 10, 3) # you should get something similar
    ```

### apex
For faster training, we need to install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Dependencies

In this project, we are using python 3.8 and PyTorch 1.10. First, let's install
the `virtualenv` tool:

- Before installing the dependencies, we have installed a virtual environment
using `virtualenv` which can be installed using:
    ```bash
    $ sudo apt-get install python3-virtualenv
    ```

- Then, we created a virtual environment called `py38` using the following
command:
    ```bash
    virtualenv py38 -p python3.8
    ```

- To activate a virtualenv, use the following command (you have to be inside
  the `true-nmt` directory):
    ```bash
    $ source py38/bin/activate
    ```
- To deactivate a virtualenv, use the following command:
    ```bash
    $ deactivate
    ```

### PyTorch

PyTorch 1.10 can be installed using the following command:
```
$ pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

If everything was installed correctly, the following should work with no
warnings or errors:
```bash
$ python
>>> import torch
>>> torch.cuda.is_available()
True

>>> torch.version.cuda
11.3 # should be the same as the install cuda driver

>>> torch.cuda.device_count()
4

>>> torch.cuda.get_device_name(0)
'NVIDIA Tesla T4'
```

### FastBPE

You can install [FastBPE](https://github.com/glample/fastBPE) for sub-work
tokenization using the following steps:

- Clone the GitHub repository:
    ```bash
    $ git clone https://github.com/glample/fastBPE.git
    ```
- Install needed dependencies:
    ```bash
    $ sudo apt-get install python3.8-dev
    $ pip install Cython
    ```
- Install FastBPE python API:
    ```bash
    $ cd fastBPE
    $ python setup.py install
    ```
- To make sure everything was install correctly, try importing it like so:
    ```bash
    $ python
    >>> import fastBPE
    ```

### PyArrow

You can install [PyArrow](https://arrow.apache.org/docs/python/index.html) using `pip` like so:
```
pip install pyarrow
```


## Tools

The following are the steps needed to install the tools used in this project

### Terashuf

Tool for shuffling big files. You can install it using the following commands:
```
git clone https://github.com/alexandres/terashuf.git
make
```

### mosesdecoder

You just need to clone it:
```bash
$ git clone https://github.com/moses-smt/mosesdecoder.git
```

### subword-nmt

You just need to clone it:
```bash
$ git clone https://github.com/rsennrich/subword-nmt.git
```

### FairSeq

To install fairseq and develop locally, follow these steps:

- Clone the repository:
    ```
    git clone https://github.com/pytorch/fairseq
    ```
- Install it:
    ```
    cd fairseq
    pip install --editable ./
    python setup.py build_ext --inplace
    ```

### Tensorboard

To install TensorBoard without TensorFlow, follow these steps:

- Install TensorBoard using the following command:

    ```bash
    $ pip install tensorboard
    ```

To access TensorBoard from your local machine, you need to follow these steps:

- Connect to the server using the following command:
    ```bash
    $ ssh -L 16006:127.0.0.1:6006 gcp1
    ```
- Get to the `true_nmt` directory:
    ```bash
    $ cd /mnt/disks/adisk/true_nmt
    ```
- Activate virtualenv:
    ```bash
    $ source py38/bin/activate
    ```
- Run TensorBoard:
    ```bash
    $ tensorboard --logdir=logs
    ```
- Now, open this URL: `http://127.0.0.1:16006/` in your browser.

# Data

## En-Fr

In this step, we are going to use the FairSeq pre-trained English-French
translation model. So, we don't need to download the data or train the model.
The model is already trained and we just need to download the pre-trained
model by following these steps:

- Download the file:
    ```bash
    $ https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
    ```
- Unzip the file:
    ```bash
    $ tar -xvf wmt14.en-fr.joined-dict.transformer.tar.bz2
    ```
- Rename it into appropriate name:
    ```bash
    $ mv wmt14.en-fr.joined-dict.transformer wmt14.en_fr
    ```
- You can use it directly now, see `nmt.py` script for details.


## Fr-En

To prepare the dataset for training/testing, follow these steps:

- Download the dataset for French-English benchmark. Running the following command
    will create a new directory called `wmt14_en_fr`:
    ```
    bash prepare-wmt14en2fr.sh
    ```
- At the end, the data stats were as follows:
    <div align="center">
        <table>
        <thead>
        <tr>
            <th></th>
            <th>train</th>
            <th>valid</th>
            <th>test</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>Before Cleaning</td>
            <td>40842333</td>
            <td>16573</td>
            <td>3003</td>
        </tr>
        <tr>
            <td>After Cleaning</td>
            <td>35789717</td>
            <td>15259</td>
            <td>3003</td>
        </tr>
        </tbody>
        </table>
    </div>


Steps for data preprocessing accroding to the model task as follows:

Unidirectional:
1) Download
2) Preprocess & normalize
3) Tokenize
4) Encode BPE
5) Clean
6) Binarize

Bidirectional:
1) Copy from unidirectional until (5)
2) Combine and add tags to the start of the sentences
2) Append our tags to the vocabulary
3) Binarize

CSW:
1) Copy from unidirectional until (3)
2) Generated CSW
3) Combine and add tags to the start of the sentences
4) Encode BPE
5) Clean
6) Shuffle
7) Binarize
---
# Train

To train the model, follow these steps:

- Running the following command will train a transformer-base model on the
  binarized data:
    ```bash
    bash train.sh
    ```
- The training will take a while, so you can check the progress using the
  following commands:
    - Show logs of training:
        ```bash
        tail -f [PATH]/training.log
        # e.g: tail -f checkpoints/transformer_base/fr_en/logs/training.log
        ```
    - Show the CPU stats:
        ```bash
        htop
        ```
    - Show the GPU stats:
        ```bash
        watch -n 1 nvidia-smi
        ```
    - Show [tensorboard logs](#tensorboard).

**NOTE:**

The following table explains every flag used for training; all flags can be found [here](https://fairseq.readthedocs.io/en/latest/command_line_tools.html) and [here](https://fairseq.readthedocs.io/en/v0.10.2/tasks.html):

<table>
<thead>
  <tr>
    <th>Flag</th>
    <th>Description</th>
    <th>Used Value</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>arch</td>
    <td>The model architecture that will be trained. This is the <a href="https://fairseq.readthedocs.io/en/latest/command_line_tools.html#Model%20configuration">total list</a> of architectures that can be used.</td>
    <td>transformer</td>
  </tr>
  <tr>
    <td>save-dir</td>
    <td>The directory to save the model checkpoints.</td>
    <td>/mnt/disks/adisk/true_nmt/checkpoints/wmt14_fr_en_transformer_base</td>
  </tr>
  <tr>
    <td>tensorboard-logdir</td>
    <td>The directory to save TensorBoard logs.</td>
    <td>/mnt/disks/adisk/true_nmt/logs</td>
  </tr>
  <tr>
    <td>optimizer</td>
    <td>The optimzier that will be used.</td>
    <td>Adam</td>
  </tr>
  <tr>
    <td>adam-betas</td>
    <td>β1 and β2 that will be used with Adam. Also, Adam has a weight-decay of 0.0001.</td>
    <td>'(0.9, 0.98)'</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Learning Rate</td>
    <td>5e-4</td>
  </tr>
  <tr>
    <td>lr-scheduler</td>
    <td>The function of time-step that will change the learning rate while training to get better performance. This scheduler has 4000 warmup-updates.</td>
    <td>inverse_sqrt</td>
  </tr>
  <tr>
    <td>dropout</td>
    <td>The dropout probability.</td>
    <td>0.1</td>
  </tr>
  <tr>
    <td>criterion</td>
    <td>The loss function.</td>
    <td>label_smoothed_cross_entropy</td>
  </tr>
  <tr>
    <td>label-smoothing</td>
    <td>The label smoothing uncertainty.</td>
    <td>0.1</td>
  </tr>
  <tr>
    <td>max-tokens</td>
    <td>Maximum number of tokens used for training. This is used instead of `batch-size`. Also, `max-tokens-valid=max-tokens` if not specified otherwise.</td>
    <td>4096</td>
  </tr>
  <tr>
    <td>num-workers</td>
    <td>The total number of parallel process that will be running while training.</td>
    <td>8</td>
  </tr>
  <tr>
    <td>validate-interval-updates</td>
    <td>The number of batches used before validation.</td>
    <td>6000</td>
  </tr>
  <tr>
    <td>task</td>
    <td>The task your model is training on. Possible choices can be found <a href="https://fairseq.readthedocs.io/en/latest/command_line_tools.html#Named%20Arguments">here</a>.</td>
    <td>translation</td>
  </tr>
  <tr>
    <td>eval-bleu</td>
    <td>Uses BLEU as the evaluation metric. This argument is usable only because the `task=translation`</td>
    <td></td>
  </tr>
  <tr>
    <td>eval-bleu-args</td>
    <td>The args that will be used with BLEU metric. All possible arguments and their default values can be found <a href="https://github.com/pytorch/fairseq/blob/main/fairseq/sequence_generator.py"> here </a>. This argument is usable only because the `task=translation`</td>
    <td>'{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'</td>
  </tr>
  <tr>
    <td>eval-bleu-deotk</td>
    <td>The algorithm/framwork that will be used for detokenization the validation set. This argument is usable only because the `task=translation`</td>
    <td>moses</td>
  </tr>
  <tr>
    <td>eval-bleu-remove-bpe</td>
    <td>Removes BPE when validating. This argument is usable only because the `task=translation`</td>
    <td></td>
  </tr>
  <tr>
    <td>eval-bleu-print-samples</td>
    <td>Print one sample per batch when validating. This argument is usable only because the `task=translation`</td>
    <td></td>
  </tr>
  <tr>
    <td>best-checkpoint-metric</td>
    <td>The metric that will be used for evaluating the best model to be saved.</td>
    <td>bleu</td>
  </tr>
  <tr>
    <td>maximize-best-checkpoint-metric</td>
    <td>Select the largest metric value for saving `best` checkpoint.</td>
    <td></td>
  </tr>
  <tr>
    <td>max-epoch</td>
    <td>The maximum number of training epochs.</td>
    <td>50</td>
  </tr>
  <tr>
    <td>patience</td>
    <td>Stop training if valid performance doesn’t improve for N validations.</td>
    <td>10</td>
  </tr>
</tbody>
</table>

During validation, the logger will print a few values under the following
acronyms:
- nll_loss: Negative log-likelihood loss.
- ppl: perplexity (the lower, the better).
- wps: Words Per Second.
- ups: Updates Per Second.
- wpb: Words Per Batch.
- bsz: batch size.
- num_updates: number of updates since the start of training.
- lr: learning rate.
- gnorm: L2 norm of the gradients.
- gb_free: GPU memory free.
- wall: total time spent training, validating, saving checkpoints (so far).
- train_wall: time taken for one training step
- oom: number of times the training was stopped because of OOM.


# Score

To evaluate your model on real data, you can use the following steps:

- Run the `score_*.sh` file like so:
  ```bash
  bash score_bidirectional.sh #for bidirectional model
  ```

During scoring, the logger will print a few lines prepended by certain characters. Here is the meaning:

- `S`: is the source sentence the model has to translate.
- `T`: is the target or the reference for the source sentence.
- `H`: is the tokenized hypothesis translation (i.e, the tokens generated by the model), along with its score.
- `D`: is the detokenized hypothesis translation (i.e, the sentence generated by the model without tokenization, in other words after applying the applied word tokenization in reverse), along with its score.
- `P`: TODO

## Useful Commands

- To get the id of the running process of `fairseq-train`, run the following
  command:
    ```bash
    ps aux | grep 'fairseq-train'
    ```
- After getting the id, kill the process by running the following command:
    ```bash
    kill -9 <id> # where <id> is the id of the process
    ```
- If that didn't work, try running the following command:
    ```bash
    for pid in $(ps -ef | awk '/python/ {print $2}'); do kill -9 $pid; done
    ```

- To stop the cronjob, just comment the line in the crontab file. You can get
the crontab by running the following command:
    ```bash
    sudo crontab -e
    ```