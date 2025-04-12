# Wyoming Haus Klaus

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [fxtentacle/wav2vec2-xls-r-1b-tevr](https://huggingface.co/fxtentacle/wav2vec2-xls-r-1b-tevr) ASR pipeline. Credits for the model goes to Krabbenhöft, Hajo Nils and Barth, Erhardt, I am not the owner of the model. I have used the rhasspy/wyoming-faster-whisper as a reference. Basically any model from HF can be used with this.

You can run this e.g. on your Desktop-PC with an RTX3060Ti. Then connect your HA Wyoming Service with your Server and thats it. Your STT request from e.g. HA Voice will then be sent to your ASR pipeline (in this case Desktop-PC), which will invoke the model and respond with an transcription of the recording to HomeAssistant.

Currently all recordings are being saved in ./data/unlabeled, might be useful for training the model later.

# Citation

```
@misc{https://doi.org/10.48550/arxiv.2206.12693,
  doi = {10.48550/ARXIV.2206.12693},
  url = {https://arxiv.org/abs/2206.12693},
  author = {Krabbenhöft, Hajo Nils and Barth, Erhardt},  
  keywords = {Computation and Language (cs.CL), Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, F.2.1; I.2.6; I.2.7},  
  title = {TEVR: Improving Speech Recognition by Token Entropy Variance Reduction},  
  publisher = {arXiv},  
  year = {2022}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/ABee81/wyoming-haus-klaus.git
cd wyoming-haus-klaus
script/setup
```

Run a server anyone can connect to:

```sh
script/run --model wav2vec2-xls-r-1b-tevr --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

The `--model` can also be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

## Docker Image

``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data rhasspy/wyoming-whisper \
    --model wav2vec2-xls-r-1b-tevr --language en
```

**NOTE**: Models are downloaded temporarily to the `HF_HUB_CACHE` directory, which defaults to `~/.cache/huggingface/hub`.
You may need to adjust this environment variable when using a read-only root filesystem (e.g., `HF_HUB_CACHE=/tmp`).

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/whisper)
