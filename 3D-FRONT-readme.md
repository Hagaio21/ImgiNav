<font class=center size=5>**3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics**</font>

## 1. Overview
<font class=center>![enter image description here](https://img.alicdn.com/tfs/TB1lRgP2xv1gK0jSZFFXXb0sXXa-3435-1085.png)</font>

We introduce [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) (3D Furnished Rooms with layOuts and semaNTics), a new, large-scale, and comprehensive repository of synthetic indoor scenes highlighted by professionally designed layouts and a large number of rooms populated by high-quality textured 3D models with style compatibility. From layout semantics down to texture details of individual objects, our dataset is freely available to the academic community and beyond. Currently, 3D-FRONT contains 18,968 rooms diversely furnished by 3D objects, far surpassing all publicly available scene datasets. In addition, the 13,151 furniture objects all come with high quality textures. While the floorplans and layout designs are directly sourced from professional creations, the interior designs in terms of furniture styles, color, and textures have been carefully curated based on a recommender system we develop to attain consistent styles as expert designs. More details can be found in our <a href='https://arxiv.org/pdf/2011.09127.pdf'>report</a>.

<br>

## 2. Data Description

3D-FRONT contains professionally and distinctively designed layouts spanning 31 scene categories, object semantics (e.g., category, style, and material labels), and a large number (18,968) of rooms populated with 3D furniture objects. Most importantly, these 3D furniture objects are all endowed with high-quality textures, thanks to [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future?spm=5176.14208320.0.0.46053cf7DVpJBh), a recently released dataset of quality 3D furniture used in industrial productions.

### 2.1. Download
The 3D-FRONT benchmark is provided by Alibaba Topping Homestyler, and organized by Alibaba Tao Technology Department. If you would like to download the 3D-FRONT data, please agree to the <a href="https://gw.alicdn.com/bao/uploaded/TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf?spm=a1z3i.a4.0.0.3f5beb1digOegr&file=TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf" >3D-FRONT Data Sets Use License Agreement</a>, and send it to us at tianchi_open_dataset@alibabacloud.com as an attachment with your name, PI's (or advisor's) name and affiliation. If you have not received a response within a week, it is likely that your email is bouncing - please check this before sending repeat requests. Please check the [news](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) for updates to the data release.

### 2.2. Data Format
The 3D-FRONT dataset includes house layouts file (3D-FRONT.zip), furniture shapes file (3D-FUTURE-model.zip), and wall / floor texture file (3D-FRONT-texture.zip).

#### 2.2.1 house layouts:
The downloaded house layouts file should have the following form:
```
3D-FRONT.zip
├── 3D-FRONT
│   ├── 3085ee0a-7da1-466e-9d5f-71ad571a25c0.json
│   ├── 87b3932e-b5b2-4539-9451-8e3abadf41b7.json
│   │...
```
* The `furniture >> jid` in json files is corresponding to the model id in downloaded furniture shapes.
* The `furniture >> valid: false` in json files represents an decoration (e.g. plants) that do not have a corresponding model. In 3D-FRONT, we also provided its bounding box in house.

#### 2.2.2 furniture shapes:
The downloaded furniture shapes file should have the following form:
```
3D-FUTURE-model.zip
├── categories.py
├── model_info.json
├── 3D-FUTURE-model
│   ├── 209fdbd6-b95d-4d11-a05e-8fcefa32a60d
│   │   ├── image.jpg
│   │   ├── normalized_model.obj
│   │   ├── raw_model.obj
│   │   ├── texture.png
│   │   ├── model.mtl
│   ├── 209fdbd6-b95d-4d11-a05e-8fcefa32a60d
│   │   ├── image.jpg
│   │   ├── normalized_model.obj
│   │   ├── raw_model.obj
│   │   ├── texture.png
│   │   ├── model.mtl
│   │...
```
* categories.py: includes furniture attributes (e.g. style, material, theme) and categories used in 3D-FRONT dataset, which are given by our experienced designers.
* model_info.json: describes the style, theme, material, and category of each furniture, except lights.

#### 2.2.3 wall and floor texture:
The downloaded wall and floor texture file should have the following form:
```
3D-FRONT-texture.zip
├── categories.py
├── texture_info.json
├── 3D-FRONT-texture
│   ├── 4e3ea8a0-9add-4934-99f8-b3bebc804e12
│   │   ├── texture.png
│   ├── 4e880ff5-8633-4925-8965-e281887c53b1
│   │   ├── texture.png
│   │...
```
* categories.py: includes texture attributes (e.g. style) and categories used in 3D-FRONT dataset, which are given by our experienced designers.
* texture_info.json: describes the style and category of each texture.

#### 2.2.4. Rendering scene images
If you would like to use the rendering scene images, please request for the [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) dataset.

<br>

## 3. Citation
If you use our dataset or code, please consider to cite our reports:
```
@article{fu20203dfront,
    title={3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics},
    author={Fu, Huan and Cai, Bowen and Gao, Lin and Zhang, Lingxiao and Li, Cao and Zeng, Qixun and Sun, Chengyue 
            and Fei, Yiyun and Zheng, Yu and Li, Ying and Liu, Yi and Liu, Peng and Ma, Lin and Weng, Le and Hu, Xiaohang 
            and Ma, Xin and Qian, Qian and Jia, Rongfei and Zhao, Binqiang and Zhang, Hao},
    journal={arXiv preprint arXiv:2011.09127},
    year={2020}
}
        

@article{fu20203dfuture,
    title={3D-FUTURE: 3D Furniture shape with TextURE},
    author={Fu, Huan and Jia, Rongfei and Gao, Lin and Gong, Mingming and Zhao, Binqiang and Maybank, Steve and Tao, Dacheng},
    journal={arXiv preprint arXiv:2009.09633},
    year={2020}
}
```