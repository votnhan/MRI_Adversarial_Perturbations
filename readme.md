## Use [Generative Adversarial Perturbation](https://arxiv.org/abs/1712.02328) method to adversarially attack brain tumor segmentation model
### Dataset: [BRATS 2018](https://www.med.upenn.edu/sbia/brats2018/data.html) 
### Process: 2 steps
![Train segmentation model h](/images/train_h.png)

![Train generator model g](/images/train_g.png)
### Result: Original data vs Noise data
**Case 1**
![Case 1](/images/comp_ori_noi_1.png)

**Case 2**
![Case 1](/images/comp_ori_noi_2.png)

### It can be seen that:
* Noise can appear globally or locally  
* Noise often appears at high intensity region. There is high probability that these region belong to the tumor
* It is difficult to distinguish between Noise and Clean data
