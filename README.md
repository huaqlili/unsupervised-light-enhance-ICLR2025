# Interpretable Unsupervised Joint Denoising and Enhancement for Real-World low-light Scenarios (ICLR2025)

[Paper] | [Openreview](https://openreview.net/forum?id=PVHoELf5UN&noteId=tWR79MUc4B)

#### News
- **Jan 22, 2025:**  Our work of unsupervised joint denoising and enhancement has been accepted to ICLR 2025!
- **Mar 17, 2025:**  We have released the pretrained model weights and testing code!


<hr />


> **Abstract:** *Real-world low-light images often suffer from complex degradations such as local overexposure, low brightness, noise, and uneven illumination. Supervised methods tend to overfit to specific scenarios, while unsupervised methods, though better at generalization, struggle to model these degradations due to the lack of reference images. To address this issue, we propose an interpretable, zero-reference joint denoising and low-light enhancement framework tailored for real-world scenarios. Our method derives a training strategy based on paired sub-images with varying illumination and noise levels, grounded in physical imaging principles and retinex theory. Additionally, we leverage the Discrete Cosine Transform (DCT) to perform frequency domain decomposition in the sRGB space, and introduce an implicit-guided hybrid representation strategy that effectively separates intricate compounded degradations. In the backbone network design, we develop retinal decomposition network guided by implicit degradation representation mechanisms. Extensive experiments demonstrate the superiority of our method.* 
>

<p align="center">
  <img width="800" src="figs/pipeline.jpg">
</p>

---

## Installation

     pyhton=3.7
     pytorch=1.11.0

## Pretrained models

We provide the Google Drive links for the following pre-trained weights.

LOLv1-test: [LOLv1_test.pth](https://drive.google.com/file/d/1UDr8zTLyfcndosGRUPOE6cgoFWAaQmwY/view?usp=drive_link)

LOLv2-test: [LOLv2_test.pth](https://drive.google.com/file/d/1deGacYmmqJ2rlwEJBnWJLWVGGbNYoyNq/view?usp=drive_link)

SICE-test: [SICE_test.pth](https://drive.google.com/file/d/1MG_90yB8HIXIfKLkNlLkQnVZgcMBmiqc/view?usp=drive_link)

You can download the LOLv1 and LOLv2 datasets via the official links. We follow the official train-test data split. For the SICE dataset, we follow the relevant settings from [PairLIE](https://github.com/zhenqifu/PairLIE).

## Testing

You can run the following code for testing：

    python eval.py --data_test path-to-data-folder \
                   --output_folder path-to-save-output

## Evaluation

You can obtain the quantitative metrics of the experiments by running the following command：

    python measure.py

## Results

<p align="center">
  <img width="800" src="figs/tab1.jpg">
</p>

<p align="center">
  <img width="800" src="figs/tab2.jpg">
</p>

## Citation
If you use DiffIR, please consider citing:

    @article{li2025Interpretable,
      title={Interpretable Unsupervised Joint Denoising and Enhancement for Real-World low-light Scenarios Unsupervised Joint Denoising and Enhancement for Real-World low-light Scenarios},
      author={Huaqiu Li and Xiaowan Hu and Haoqian Wang},
      journal={ICLR},
      year={2025}
    }

Our work is built upon the codebase of [PairLIE](https://github.com/zhenqifu/PairLIE), and we sincerely thank them for their contribution.
