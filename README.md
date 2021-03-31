# An Image is Worth 16x16 Words, What is a Video Worth?

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-image-is-worth-16x16-words-what-is-a-video/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=an-image-is-worth-16x16-words-what-is-a-video)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-image-is-worth-16x16-words-what-is-a-video/action-recognition-in-videos-on-ucf101)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ucf101?p=an-image-is-worth-16x16-words-what-is-a-video)

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
    <td><b>76.0</b></td>
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
    <td><b>48.4 × 30</b></td>
    <td><b>480</b></td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>STAM-16</td>
    <td>77.8</td>
    <td>270 × 1</td>
    <td>16</td>
    <td>20.0</td>
  </tr>
  <tr>
    <td>STAM-64</td>
    <td>79.2</td>
    <td>1080 × 1</td>
    <td>64</td>
    <td><b>4.8</b></td>
  </tr>
 </table>
</p>

## Pretrained Models

Coming soon

## Reproduce Article Scores
We provide code for reproducing the validation top-1 score of STAM
models on Kinetics400. First, download pretrained models from the links above.

Then, run the infer.py script. For example, for stam_16 (input size 224)
run:
```bash
python -m infer.py \
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
