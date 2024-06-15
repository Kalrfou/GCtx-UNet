# GCxt-UNet: Efficient Deep Network for Medical Image Segmentation

#### GCtx-UNet is a U-shaped network architecture that incorporates the Global Context Vision Transformer (GC-ViT) to enhance medical image segmentation by effectively capturing both global and local features. Feel free to check out our preprint on [arXiv](https://arxiv.org/pdf/2406.05891). &#8291;
##### The complete code will be published soon.
## Pre-trained model:
#### 1.  Download pre-trained GC-ViT transformer model (GCViT-xxtiny) pre-trained on ImageNet1K 
   [Get pre-trained model in this link] (https://drive.usercontent.google.com/download?id=1apSIWQCa5VhWLJws8ugMTuyKzyayw4Eh&export=download&authuser=0): Put pretrained xx-Tiny into folder "pretrained_ckpt/"
#### 2.  Download pre-trained GC-ViT transformer model (GCViT-xxtiny) pre-trained on MedNet 

##  Prepare data
The datasets we used are provided by the authors of TransUnet. You can access the processed data through this [link](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). For more details, please refer to the "./datasets/README.md" file. Alternatively, you can go ahead and request the preprocessed data by emailing jienengchen01@gmail.com. If you use the preprocessed data, please ensure it is solely for research purposes and do not redistribute it by TransUnet's License.

## Testing
#### Get pre-trained GCtx-UNet model weights on the Synapse dataset:  [link](https://panthers-my.sharepoint.com/:u:/g/personal/tzhao_uwm_edu/ER6J2LwtirFOip2m6u7hQt8BBdph8P2OrfI_Wmj8MNMQfg?e=7uiftc)
download the file and put it into the folder model_out.

```bash
CUDA_VISIBLE_DEVICES=0  python -W ignore test.py
```
## References
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [GC-ViT](https://github.com/NVlabs/GCVit)
  
## Citation

```bibtex
@article{alrfou2024gctx,
  title={GCtx-UNet: Efficient Network for Medical Image Segmentation},
  author={Alrfou, Khaled and Zhao, Tian},
  journal={arXiv preprint arXiv:2406.05891},
  year={2024}
}
