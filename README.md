![image](https://github.com/user-attachments/assets/5ca59498-8f75-469c-8546-c9b20f9593ea)
![image](https://github.com/user-attachments/assets/08617ad9-b9a4-49e2-90fc-3429ac60f9b2)

### **0.  Create Conda Virtual Environment (Ubuntu 22.04)**

  `conda create -n sam2 python=3.13 -y`
  
  `conda activate sam2`


### **1.  SAM2 Installation**

If you already deploy SAM2 on your workstation, move to next step.

    `git clone https://github.com/facebookresearch/sam2.git && cd sam2`

    `pip install -e .`

For more details, following the instruction of https://github.com/facebookresearch/sam2

### **2.  Clone ISSAS into the root path of SAM2**

  `git clone https://github.com/mqxwd68/ISSAS.git && cd ISSAS`
