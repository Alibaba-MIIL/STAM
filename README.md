# An Image is Worth 16x16 Words, What is a Video Worth?

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-image-is-worth-16x16-words-what-is-a-video/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=an-image-is-worth-16x16-words-what-is-a-video) -->

[paper](https://arxiv.org/pdf/2103.13915.pdf) 

Official PyTorch Implementation

> Gilad Sharir, Asaf Noy, Lihi Zelnik-Manor<br/>
> DAMO Academy, Alibaba Group



**Abstract**

> Leading methods in the domain of action recognition try to
distill information from both the spatial and temporal dimensions of an input video. Methods that reach State of the
Art (SotA) accuracy, usually make use of 3D convolution
layers as a way to abstract the temporal information from
video frames. The use of such convolutions requires sampling short clips from the input video, where each clip is a
collection of closely sampled frames. Since each short clip
covers a small fraction of an input video, multiple clips are
sampled at inference in order to cover the whole temporal
length of the video. This leads to increased computational
load and is impractical for real-world applications. We address the computational bottleneck by significantly reducing
the number of frames required for inference. Our approach
relies on a temporal transformer that applies global attention over video frames, and thus better exploits the salient
information in each frame. Therefore our approach is very
input efficient, and can achieve SotA results (on Kinetics
dataset) with a fraction of the data (frames per video), computation and latency. Specifically on Kinetics-400, we reach
78.8 top-1 accuracy with ×30 less frames per video, and
×40 faster inference than the current leading method
>

## Update 2/5/2021:  Improved results
Due to improved training hyperparameters, and using KD training, we were able to improve
 STAM results on Kinetics400 (+ ~1.5%).  We are releasing the pretrained weights of the improved
  models (see Pretrained Models below). 

## Main Article Results

STAM models accuracy and GPU throughput on Kinetics400, compared to X3D. All measurements were
 done on Nvidia V100 GPU, with mixed precision. All models are trained on input resolution of 224.
<p align="center">
 <table>
  <tr>
    <th>Models</th>
    <th>Top-1 Accuracy <br>(%)</th>
    <th>Flops × views<br>(10^9)</th>
    <th># Input Frames</th>
    <th>Runtime<br>(Videos/sec)</th>
  </tr>
  <tr>
    <td>X3D-M</td>
    <td>76.0</td>
    <td>6.2 × 30 </td>
    <td>480</td>
    <td>1.3</td>
  </tr>
  <tr>
    <td>X3D-L</td>
    <td>77.5</td>
    <td>24.8 × 30</td>
    <td>480</td>
    <td>0.46</td>
  </tr>
  <tr>
    <td>X3D-XL</td>
    <td>79.1</td>
    <td>48.4 × 30</td>
    <td>480</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>X3D-XXL</td>
    <td>80.4</td>
    <td>194 × 30</td>
    <td>480</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>TimeSformer-L</td>
    <td>80.7</td>
    <td>2380 × 3</td>
    <td> 288 </td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>ViViT-L</td>
    <td><b>81.3</b></td>
    <td>3992 × 12</td>
    <td>384</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>STAM-8</td>
    <td>77.5</td>
    <td><b>135 × 1</b></td>
    <td><b>8</b></td>
    <td><b>---</b></td>
  </tr>
  <tr>
    <td>STAM-16</td>
    <td>79.3</td>
    <td><b>270 × 1</b></td>
    <td><b>16</b></td>
    <td><b>20.0</b></td>
  </tr>
  <tr>
    <td>STAM-32</td>
    <td>79.95</td>
    <td><b>540 × 1</b></td>
    <td><b>32</b></td>
    <td><b>---</b></td>
  </tr>
  <tr>
    <td>STAM-64</td>
    <td><b>80.5</b></td>
    <td><b>1080 × 1</b></td>
    <td>64</td>
    <td>4.8</td>
  </tr>
 </table>
</p>

## Pretrained Models

We provide a collection of STAM models pre-trained on Kinetics400. 

| Model name  | checkpoint
| ------------ | :--------------: |
| STAM_8 | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/STAM/v2/stam_8.pth) |
| STAM_16 | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/STAM/v2/stam_16.pth) |
| STAM_32 | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/STAM/v2/stam_32.pth) |
| STAM_64 | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/STAM/v2/stam_64.pth) |


## Reproduce Article Scores
We provide code for reproducing the validation top-1 score of STAM
models on Kinetics400. First, download pretrained models from the links above.

Then, run the infer.py script. For example, for stam_16 (input size 224)
run:
```bash
python -m infer \
--val_dir=/path/to/kinetics_val_folder \
--model_path=/model/path/to/stam_16.pth \
--model_name=stam_16
--input_size=224
```


## Citations

```bibtex
@misc{sharir2021image,
    title   = {An Image is Worth 16x16 Words, What is a Video Worth?}, 
    author  = {Gilad Sharir and Asaf Noy and Lihi Zelnik-Manor},
    year    = {2021},
    eprint  = {2103.13915},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

## Acknowledgements

We thank Tal Ridnik for discussions and comments.

Some components of this code implementation are adapted from the excellent
[repository of Ross Wightman](https://github.com/rwightman/pytorch-image-models). Check it out and give it a star while
you are at it.
