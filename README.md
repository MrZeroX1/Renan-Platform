
<div align="center">

[![English](https://img.shields.io/badge/Language-English-blue)](#english) 
[![Arabic](https://img.shields.io/badge/Language-Arabic-green)](#arabic)
</div>

<div align="center">


## <img src="assest/White_Logo.png" height="200"/>


**Fine-Tuning TTS Model For Arabic Language**

<audio controls>
  <source src="assest/synthesized_audio.mp3" type="audio/mpeg">
</audio>

</div>


## Team Members
- [Sultan AlBayyat - سلطان البيات](https://www.linkedin.com/in/sultan-albayyat/)
- [Fatimah Aljishi - فاطمة الجشي](https://www.linkedin.com/in/fatimah-aljishi-103927291/)
- [Hasan Alzayer - حسن الزاير](https://www.linkedin.com/in/hasan-alzayer/)
- [Abdullah Al-Tamh - عبدالله الطعمة](https://www.linkedin.com/in/abdullah-al-tamh-643851281/)
- [Sarah Alshaikhmohammed - ساره آل شيخ محمد](https://www.linkedin.com/in/sarah-alshaikhmohammed-ab20a9252/)



## Features
- High-performance Deep Learning xtts model for Text2Speech tasks.
    - Speaker Encoder to compute speaker embeddings efficiently.
- Fast and efficient model training.
- Detailed training logs on the terminal and Tensorboard.
- Support for Multi-speaker TTS.
- Efficient, flexible, lightweight but feature complete `Trainer API`.
- Released and ready-to-use models.
- Utilities to use and test your models.
- Modular (but not too much) code base enabling easy implementation of new ideas.

## Prequests

- `3.9.x <= Python < 3.12`
- `CUDA >= 11.8`

## Installation
Follow these steps for installation:

1. Ensure that `CUDA` is installed
2. Clone the repository: 

```bash
git clone https://github.com/Haurrus/xtts-trainer-no-ui-auto
```

3. Navigate into the directory: 

``` bash
cd xtts-trainer-no-ui-auto
```
4. Create a virtual environment: 
    - On Terminal 
    ``` bash
    python3.11 -m venv venv
    ```
    - On Anaconda 
    ``` bash
    conda create --name myenv python=3.11.9
    ```
5. Activate the virtual environment:
   - On Anaconda : 
   ```bash
   conda activate myenv
   ```
   - On Windows use : 
   ```bash
   venv\scripts\activate
   ```
   - On linux use: 
   ```bash
   source venv\bin\activate
   ```

6. Ensure that you install `CUDA Toolkit 12.4` from their official site [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) (I choose 12.4 not 12.5 because the PyTorch only support 12.4 for now)


7. Install PyTorch and torchaudio with pip command :

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

8. Install all dependencies from requirements.txt :

    ```bash
    pip install -r requirements.txt
    ```


## During the implementation

This is a Python script for fine-tuning a text-to-speech (TTS) model for xTTSv2. The script utilizes custom datasets and use CUDA for accelerated training.

To use the script, you need to specify two JSON files: `args.json` and `datasets.json`.

### `args.json`

```json
{
  "num_epochs": 0,
  "batch_size": 3,
  "grad_acumm": 84,
  "max_audio_length": 15,
  "language": "ar",
  "version": "",
  "json_file": "",
  "custom_model": ""
}
```
This file should contain the following key parameters:
- `num_epochs`: Number of epochs for training, if set to 0 it will auto calculate it.
- `batch_size`: Batch size for training.
- `grad_acumm`: Gradient accumulation steps.
- `max_audio_length`: max audio duration of wavs used to train.
- `language`: language used to train the model.
- `version`: by default main from xTTSv2
- `json_file`: by default main from xTTSv2
- `custom_model`: by default main from xTTSv2

### `datasets.json`

```json
[
  {
    "path": "path/to/dataset1",
    "activate": true
  },
  {
    "path": "path/to/dataset2",
    "activate": false
  }
]
```

This file should list the datasets to be used with paths and activation flags.

### Running the Script
Execute the script with the following command:
```bash
python xtts_finetune_no_ui_auto.py --args_json path/to/args.json --datasets_json path/to/datasets.json
```

If you running the script with the same folder for `args.json` & `datasets.json`, run this command: 
```bash
python xtts_finetune_no_ui_auto.py --args_json args.json --datasets_json datasets.json
```
## Error handling

This section addresses some of the errors encountered while trying to execute the code.


### `torch.OutOfMemoryError`
```bash
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 6.00 GiB of which 0 bytes is free. Of the allocated memory 8.38 GiB is allocated by PyTorch, and 471.68 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
```
<br>
Potential Causes:

- Lack of Memory: The minimum requirement for running the code is 16 GB of RAM. Upgrade your device or find another device with sufficient memory.

- Lack of GPU: Ensure that you have a GPU with at least 4 GB of VRAM, as deep learning training requires significant GPU resources.

### `Access is deniend`
```bash
[WinError 5] Access is denied:'path\\to\\dataset\\run'
```
Remove the `run` folder from the dataset.

### `OSError: [WinError 1455]`
```
OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading ".root\.conda\envs\py11\Lib\site-packages\torch\lib\cufft64_11.dll" or one of its dependencies.
```
<br>
This error is similar to torch.OutOfMemoryError and can be resolved by addressing memory issues. <br>

### `Future Warning`
These warnings can be safely ignored.

## What is the Output of the Training
```
PS C:\xtts-trainer-no-ui-auto> python xtts_finetune_no_ui_auto.py --args_json args.json --datasets_json datasets.json

Checking dataset in path: C:\xtts-trainer-no-ui-auto\outwaves
Looking for dataset at: C:\xtts-trainer-no-ui-auto\outwaves
 > Loading custom model: C:\xtts-trainer-no-ui-auto\models\main\model.pth

>> DVAE weights restored from: C:\xtts-trainer-no-ui-auto\models\main\dvae.pth
 | > Found 40 files in 

 > Training Environment:
 | > Backend: Torch
 | > Mixed precision: False
 | > Precision: float32
 | > Num. of CPUs: 12
 | > Num. of GPUs: 1
 | > Num. of Torch Threads: 1
 | > Torch seed: 1
 | > Torch CUDNN: True
 | > Torch CUDNN deterministic: False
 | > Torch CUDNN benchmark: False
 | > Torch TF32 MatMul: False
 > Start Tensorboard: tensorboard --logdir=C:\xtts-trainer-no-ui-auto\outwaves\run\training\GPT_XTTS_FT-August-05-2024_05+14PM-8025ef4

 > Model has 518442047 parameters

 > EPOCH: 0/50
 --> C:\xtts-trainer-no-ui-auto\outwaves\run\training\GPT_XTTS_FT-August-05-2024_05+14PM-8025ef4
 > Filtering invalid eval samples!!
 > Total eval samples after filtering: 40

 > EVALUATION 


  --> EVAL PERFORMANCE
     | > avg_loader_time: 0.18054273189642486 (+0)
     | > avg_loss_text_ce: 0.033357221107834435 (+0)
     | > avg_loss_mel_ce: 4.670955278934577 (+0)
     | > avg_loss: 4.704312489582942 (+0)


 > EPOCH: 1/100
 --> C:\xtts-trainer-no-ui-auto\outwaves\run\training\GPT_XTTS_FT-August-05-2024_05+14PM-8025ef4
 > Sampling by language: dict_keys(['ar'])

 > TRAINING (2024-08-05 17:17:08) 
 --> TIME: 2024-08-03 02:51:47 -- STEP: 0/14 -- GLOBAL_STEP: 0[0m
     | > loss_text_ce: 0.0332101508975029  (0.0332101508975029)
     | > loss_mel_ce: 4.7347731590271  (4.7347731590271)
     | > loss: 0.05676170811057091  (0.05676170811057091)
     | > current_lr: 5e-06 
     | > step_time: 0.9667  (0.9666781425476074)
     | > loader_time: 79.179  (79.178950548172)


  > EVALUATION


--> EVAL PERFORMANCE
     | > avg_loader_time: 0.028109293717604417 (+0.0017166871290940494)
     | > avg_loss_text_ce: 0.032951588957355574 (-7.596425712108612e-05)
     | > avg_loss_mel_ce: 4.6023083833547735 (-0.07368043752817055)
     | > avg_loss: 4.635259958413931 (-0.07375643803523246)

 > BEST MODEL : C:\xtts-trainer-no-ui-auto\outwaves\run\training\GPT_XTTS_FT-August-05-2024_05+14PM-8025ef4\best_model_14.pth

 > EPOCH: 2/100
 -->  C:\xtts-trainer-no-ui-auto\outwaves\run\training\GPT_XTTS_FT-August-05-2024_05+14PM-8025ef4

TRAINING (2024-08-03 02:52:07)
```


# <strong>Warning </strong>: This project is not developed by us; we are merely using it to train the model for Najdi and Khaliji dialects support.

### All thanks to [Haurrus/xtts-trainer-no-ui-auto](https://github.com/Haurrus/xtts-trainer-no-ui-auto/) for the works he done on this project.
