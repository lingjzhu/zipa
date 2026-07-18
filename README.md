# ZIPA: A family of efficient speech models for multilingual phone recognition - ACL 2025

Read [the ZIPA paper](https://aclanthology.org/2025.acl-long.961/) here!

**[Update]**: Clean labels for the IPA Pack Plus is available at [https://huggingface.co/datasets/sejongwang/ipapack_plus_clean](https://huggingface.co/datasets/sejongwang/ipapack_plus_clean). This is a label only repo. Thanks Junehwi Kim at KAIST for cleaning the data!! Please cite his repository if you use his clean labels. 

## ONNX Inference
For users who are only interested in running the model for inference, we provide optimized ONNX models (FP32, FP16, and INT8) for efficient inference (on your browsers and phones). We have included checkpoints in `fp32`, `fp16` and `int8` in the `Final Averaged Checkpoint` HF hubs below. The dependencies are reduced to minimal to facilitate usage. Note that low precision models might lead to slightly worse performance, despite the gain of efficiency. Sample results are [here](https://github.com/lingjzhu/zipa/blob/main/scripts/onnx_scripts/inference_results_all.txt).
**Thanks @guettli for the suggestion!**

### Setup
1. Install dependencies:
   ```bash
   pip install onnxruntime soundfile librosa lhotse torch
   ```
2. Download exported models from HF hubs. A list of models is available [here](https://github.com/lingjzhu/zipa/blob/main/README.md#pretrained-models). 

### Usage
Use the simplified inference script in `inference/inference.py`.

#### CTC Models
```bash
python inference/inference.py path/to/audio.wav --model-path checkpoints/zipa-cr-small-300k/exp/model.onnx --model-type ctc
```

#### Transducer Models
For Transducer, pass the **directory** containing the ONNX files (`encoder`, `decoder`, `joiner`).
```bash
python inference/inference.py path/to/audio.wav --model-path checkpoints/zipa-t-small-300k/exp --model-type transducer
```

#### Optional Arguments
- `--tokens`: Path to `tokens.txt` (default: `ipa_simplified/tokens.txt`).
- `--suffix`: Suffix for Transducer files, e.g., `.fp16.onnx` or `.int8.onnx` (default: `.onnx`).

### Batch Inference
To process a directory of audio files (wav, flac, mp3), use `inference/batch_inference.py`.

```bash
python inference/batch_inference.py path/to/audio_dir --model-path checkpoints/zipa-cr-small-300k/exp/model.onnx --model-type ctc --batch-size 32
```

Optional arguments:
- `--batch-size`: Batch size (default: 1).
- `--tokens`: Path to `tokens.txt`.


### Example
```bash
# Run FP16 Transducer inference
python inference/inference.py sample.wav --model-path checkpoints/zipa-t-small-300k/exp --model-type transducer --suffix .fp16.onnx
```

# Train and evaluate models with `torch`
This repo is built upon the [Icefall](https://github.com/k2-fsa/icefall) library in Next-gen Kaldi. The usage is almost the same. 


## Environment
Please refer to `icefall_container.def` for a complete setup of the environment.

A pre-built apptainer container can be found [here](https://huggingface.co/datasets/anyspeech/ipapack_plus_meta/blob/main/pytorch2.4.0-cuda12.4-icefall-container.sif).

You can build an apptainer (which works without root access on HPC) with the given definition file.
```
apptainer build icefall.sif icefall_container.def
```

Generally speaking, packages below are required for minimal usage:
  1. `torch torchaudio torchvision`
  2. `lhotse` (for audio preprocessing)
  3. `icefall` and `k2`. They must exactly match the torch version and cuda version. Instructions are available [here](https://icefall.readthedocs.io/en/latest/installation/index.html).
  4. `huggingface_hub` (for downloading models and data)
  5. Optional: `kaldifeat`. If you need to train from scratch, this library is also required. See instructions [here](https://csukuangfj.github.io/kaldifeat/installation/from_wheels.html). It must match the torch and cuda versions strictly.

Run `export PYTHONPATH=/icefall:$PYTHONPATH` if the apptainer cannot find `icefall`.
     
# Inference

## Batch inference with detailed error logs
You might need to modify some paths in the `data_module.py` to point to your local data. 
```  
python zipformer_crctc/ctc_decode.py --iter 800000 --avg 10 --exp-dir /scratch/lingjzhu_root/lingjzhu1/lingjzhu/zipformer_exp/zipformer_large_crctc_75_pretrained  \
--use-transducer False --use-ctc True  --use-cr-ctc True --max-duration 600 --decoding-method ctc-greedy-search \
--bpe-model ipa_simplified/unigram_127.model --num-workers 1 --num-encoder-layers 4,3,4,5,4,4 \
--feedforward-dim 768,768,1536,2048,1536,768 --encoder-dim 512,512,768,1024,768,512 \
--encoder-unmasked-dim 192,192,256,320,256,192 --decoder-dim 1024 --joiner-dim 1024 \
--query-head-dim 64 --value-head-dim 48 --num-heads 6,6,6,8,6,6 

```

## Simple inference

Please check out `zipa_ctc_inference.py` and `zipa_transducer_inference.py` for example usage.

Here are some simple instructions:
 1. Download models from Huggingface Hub (see Final Averaged Checkpoint column below). Use `zipa_ctc_inference.py` for CTCTC models and `zipa_transducer_inference.py` for transducer models.
 2. Perform inference. You can directly pass a list of audio arrays. Batching and padding are supported. Greedy decoding is used for all models. 
    - CRCTC Model
    ```python
    import torchaudio
    from zipa_ctc_inference import initialize_model
    
    # specify the path to model weights and tokenizers
    model_path = "zipformer_weights/zipa_large_crctc_500000_avg10.pth"
    bpe_model_path = "ipa_simplified/unigram_127.model"

    # initialize model
    model = initialize_model(model_path, bpe_model_path)

    # Generate a dummy audio batch (3 samples of 2 seconds of silence)
    # You can pass a list of audio arrays with any length.
    # Batching, padding, and unpadding will be handled by the code. 
    sample_rate = 16000
    dummy_audio = [torch.zeros(int(sample_rate * 2)),
                   torch.zeros(int(sample_rate * 2)),
                   torch.zeros(int(sample_rate * 2))] 

    # Run inference
    output = model.inference(dummy_audio)
    print("Predicted transcript:", output) # A list of predicted phone sequence. 
    ``` 

    - Transducer model
    ```python
    import torchaudio
    from zipa_transducer_inference import initialize_model
    
    model_path = "zipformer_weights/zipa_large_noncausal_500000_avg10.pth"
    bpe_model_path = "ipa_simplified/unigram_127.model"

    model = initialize_model(model_path, bpe_model_path)

    # Generate a dummy audio batch (3 sample of 2 seconds of silence)
    sample_rate = 16000
    dummy_audio = [torch.zeros(int(sample_rate * 2)),
                   torch.zeros(int(sample_rate * 2)),
                   torch.zeros(int(sample_rate * 2))]  

    # Run inference
    output = model.inference(dummy_audio)
    print("Predicted transcript:", output)
    
    ```
### Pretrained models
The huggingface page contains the last 10 checkpoints. The inference code will average across 10 checkpoints to make inference. 
After you download checkpoints to your local folder, you can use the inference code. `--exp-dir` should point to your local checkpoint folders.
`--iter` should be the last iteration as specified in the checkpoint names. `--avg 10` implies that the last 10 checkpoints will be averaged. Please don't change this argument, as we have only provided the last 10 checkpoints. 

| Model               | Params | Training Steps | Raw Checkpoints | Final Averaged Checkpoint |   
|---------------------|--------|--------|-------------|-------------|
| Zipa-T-small        | 65M    | 300k   | [link](https://huggingface.co/anyspeech/zipa-t-s)        | [anyspeech/zipa-small-noncausal-300k](https://huggingface.co/anyspeech/zipa-small-noncausal-300k) |
| Zipa-T-large        | 302M   | 300k   | [link](https://huggingface.co/anyspeech/zipa-t-l)        | [anyspeech/zipa-large-noncausal-300k](https://huggingface.co/anyspeech/zipa-large-noncausal-300k) |
| Zipa-T-small        | 65M    | 500k   | [link](https://huggingface.co/anyspeech/zipa-t-s)        | [anyspeech/zipa-small-noncausal-500k](https://huggingface.co/anyspeech/zipa-small-noncausal-500k) |
| Zipa-T-large        | 302M   | 500k   | [link](https://huggingface.co/anyspeech/zipa-t-l)        | [anyspeech/zipa-large-noncausal-500k](https://huggingface.co/anyspeech/zipa-large-noncausal-500k) |
| Zipa-Cr-small       | 64M    | 300k   | [link](https://huggingface.co/anyspeech/zipa-cr-s/tree/main)        | [anyspeech/zipa-small-crctc-300k](https://huggingface.co/anyspeech/zipa-small-crctc-300k) |
| Zipa-Cr-large       | 300M   | 300k   | [link](https://huggingface.co/anyspeech/zipa-cr-l)        | [anyspeech/zipa-large-crctc-300k](https://huggingface.co/anyspeech/zipa-large-crctc-300k) |
| Zipa-Cr-small       | 64M    | 500k   | [link](https://huggingface.co/anyspeech/zipa-cr-s/tree/main)        | [anyspeech/zipa-small-crctc-500k](https://huggingface.co/anyspeech/zipa-small-crctc-500k) |
| Zipa-Cr-large       | 300M   | 500k   | [link](https://huggingface.co/anyspeech/zipa-cr-l)        | [anyspeech/zipa-large-crctc-500k](https://huggingface.co/anyspeech/zipa-large-crctc-500k) | 
| Zipa-Cr-Ns-small    | 64M    | 700k   | [link](https://huggingface.co/anyspeech/zipa-cr-ns-s)        | [anyspeech/zipa-small-crctc-ns-700k](https://huggingface.co/anyspeech/zipa-small-crctc-ns-700k) |
| Zipa-Cr-Ns-large    | 300M   | 800k   | [link](https://huggingface.co/anyspeech/zipa-cr-ns-l)        | [anyspeech/zipa-large-crctc-ns-800k](https://huggingface.co/anyspeech/zipa-large-crctc-ns-800k) | 
| *No diacritics*  |        |        |             |
| Zipa-Cr-Ns-small    | 64M    | 700k   | [link](https://huggingface.co/anyspeech/zipa-cr-ns-s-no-diacritics/tree/main)        | [anyspeech/zipa-small-crctc-ns-no-diacritics-700k](https://huggingface.co/anyspeech/zipa-small-crctc-ns-no-diacritics-700k) | 
| Zipa-Cr-Ns-large    | 300M   | 780k   | [link](https://huggingface.co/anyspeech/zipa-cr-ns-l-no-diacritics)        | [anyspeech/zipa-large-crctc-ns-no-diacritics-780k](https://huggingface.co/anyspeech/zipa-large-crctc-ns-no-diacritics-780k) | 




## Data

**[Update]**: Clean labels for this data is available at [https://huggingface.co/datasets/sejongwang/ipapack_plus_clean](https://huggingface.co/datasets/sejongwang/ipapack_plus_clean). This is a label only repo. Thanks Junehwi Kim at KAIST for cleaning the data!! Please cite his repository if you use his clean labels. 

The tokenizer can be found [here](https://huggingface.co/datasets/anyspeech/ipapack_plus_meta). You'll need the `sentencepiece` package to load it. [This](https://huggingface.co/datasets/anyspeech/ipapack_plus_meta/blob/main/ipa_simplified/unigram_127.vocab) is the list of selected IPA symbols. 

All data are distributed in the scalable `shar` format, similar to `webdataset` format but with indexes. It can be easily loaded with `lhotse` library. Audio files are downsampled to 16000Hz and stored in the `flac` format to save space. 

 - [All processed data](https://huggingface.co/collections/anyspeech/ipapack-raw-673c2d345deec72e82e28a3b) (~1.8TB)
 - [Training data only](https://huggingface.co/collections/anyspeech/ipa-pack-train-6838a6804a3a71a91794a801) (~1.5TB)
 - [Pseudolabeled data](https://huggingface.co/collections/anyspeech/ipa-pack-train-pseudolabel-6838a6adc3ccad443cfb63b0) (~1TB)

After downloading all data, place all `tar` and `json` files within the same folder. 

```
data-shar
├── cuts.000000.jsonl.gz
├── recording.000000.tar
```
Then you can construct a data loader with `lhotse`. Please refer to the `lhotse` documentation and [their shar tutorial](https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb) for further details. 
```python
cuts_full = CutSet.from_shar(
    fields={
        "cuts": ["data-shar/cuts.000000.jsonl.gz"],
        "recording": ["data-shar/recording.000000.tar"],
    }
)
```

## Training
You might need to modify some paths in the `data_module.py` to point to your local data. 
Training a Zipformer-Large CRCTC model

```
python zipformer_crctc/train.py --world-size 2 --num-epochs 2 --start-epoch 1 --start-batch 500000  \
--use-fp16 1 --exp-dir /lustre07/scratch/lingjzhu/zipformer_exp/zipformer_large_crctc_0.5_scale --causal 0 \
--full-libri True --max-duration 120 --use-transducer False --use-ctc True  --use-cr-ctc True --base-lr 0.015 \
 --enable-spec-aug False --seed 2333 --wandb False --num-encoder-layers 4,3,4,5,4,4 \
--feedforward-dim 768,768,1536,2048,1536,768 --encoder-dim 512,512,768,1024,768,512 \
--encoder-unmasked-dim 192,192,256,320,256,192 --decoder-dim 1024 --joiner-dim 1024 \
--query-head-dim 64 --value-head-dim 48 --num-heads 6,6,6,8,6,6 --num-buckets 8 --num-workers 4 \
--unsup-cr-ctc-loss-scale 0.5 --use-unsup-cr-ctc True
```

Remove diacritics
```
python zipformer_crctc/train.py --world-size 2 --num-epochs 2 --start-epoch 1 --start-batch 500000 --use-fp16 1 \
 --exp-dir /lustre07/scratch/lingjzhu/zipformer_exp/zipformer_large_crctc_0.5_scale_no_diacritics --causal 0 \
--full-libri True --max-duration 120 --use-transducer False --use-ctc True  --use-cr-ctc True --base-lr 0.015 \
 --enable-spec-aug False --seed 2333 --wandb False --num-encoder-layers 4,3,4,5,4,4 \
--feedforward-dim 768,768,1536,2048,1536,768 --encoder-dim 512,512,768,1024,768,512 --encoder-unmasked-dim 192,192,256,320,256,192 \
--decoder-dim 1024 --joiner-dim 1024 --query-head-dim 64 --value-head-dim 48 --num-heads 6,6,6,8,6,6 \
--num-buckets 8 --num-workers 4 --unsup-cr-ctc-loss-scale 0.5 --use-unsup-cr-ctc True --remove-diacritics True
```

