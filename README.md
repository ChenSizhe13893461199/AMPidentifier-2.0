### Brief Introduction of Developers
#### Developer Introduction

**Sizhe Chen**, PhD Student (Medical Sciences) at Chinese University of Hong Kong (1155202847@link.cuhk.edu.hk). Supervisor: **Professor CHAN Ka Leung Francis**，**Professor Siew N.G.** and **Research Assistant Professor Qi Su**. Welcome to contact **Sizhe Chen** via the aforementioned email if you have any questions or suggestions.

This work is supervised by **Professor Siew N.G.**, **Professor Yang Sun** and **Research Assistant Professor Qi Su**. The research work is primarily finished by **Sizhe Chen** (PhD student), and **Yuan Yue** (MPhil student) with equal contributions.

# AMPidentifier-2.0
This is an updated version of AMPidentifier 1.0 (https://github.com/ChenSizhe13893461199/Fast-AMPs-Discovery-Projects), with a rapid training rate on a normal laptop and overall high performances.

For more details on AMPidentifier 1.0, please refer to our previously published article: 

[1] **Sizhe Chen#**, Huitang Qi#, Xingzhuo Zhu#, Tianxiang Liu, Yutin Fan, Qi Su, Qiuyu Gong*, Cangzhi Jia*, Tian Liu*. Screening and identification of antimicrobial peptides from the gut microbiome of cockroach _Blattella germanica_. **_Microbiome_** 2024, 12: 272. IF=16.837 doi: **10.1186/s40168-024-01985-9**.

### The framework of AMPidentifier 2.0:
![](Framework.png)
The source codes of the AMPidentifier 2.0 are available here, with an average **AUPRC indicator of 0.9486±0.0003** and a significantly reduced training time of **3200±53s** on the normal laptop (Intel i7-10875H CPU). Compared to the previous model AMPidentifier 1.0 (**AUPRC: 0.9495±0.0022**, **training time: 15,374±169s**), with overall fitting parameters and time costs decreased by approximately 56% and 80%, respectively. With the introduction of three computational modules, it still showed high overall prediction performance and low false-positive conditions (**Specificity: 90.1347±0.9487%**, **Sensitivity: 99.6864±0.046%**).

### Details of Using AMPidentifier 2.0
### Requirements
- python 3.9.7 or higher
- keras==2.10.0
- pandas==1.5.2
- matplotlib==3.0.3
- propy3 (tutorial: https://propy3.readthedocs.io/en/latest/UserGuide.html)
- numpy==1.23.5
- sklearn=1.2.0
- propy3=1.1.0
- gensim=4.2.0

**1.** The deployment codes can be found in "Train_AMP_identifier2.py", with full details and annotations attached. Firstly, open "Train_AMP_identifier2.py" on your laptop (e.g. by Spyder) and introduce all the necessary packages aforementioned.

**2.** Prepare the necessary dataset (training dataset, validation dataset, and test dataset), and their descriptors (for one-hot code, and word2-vec, the necessary documents have been provided in the GitHub repository). For the physiochemical descriptors, they are not uploaded to the GitHub repository because of their large size. You can use the codes and annotations provided in "Train_AMP_identifier2.py" (lines 81-86) to calculate physiochemical descriptors. Alternatively, you can also contact Sizhe Chen (1155202847@link.cuhk.edu.hk) to request these documents.

**3.** Following the codes provided in "Train_AMP_identifier2.py" and running it in your local Spyder or other suitable environment directly, the training of AMPidentifier 2.0 will be smoothly performed.

**Notes:**

Some available laptops may not be able to operate AMPidentifier 2.0 because of memory limitations. In that condition, you can copy the codes and run AMPidentifier 2.0 on the Jupyter viewer deployed on the high-performance server (e.g. CPU: Xeon(R) Gold 6430, GPU: RTX 4090, RAM: 120GB).

We are preparing a detailed tutorial for both AMPidentifier 1.0 and AMPidentifier 2.0 (will be available soon). Please feel free to contact us if you have any suggestions. Thank you for your attention.

The model core file is also being prepared and will be uploaded to Github soon
