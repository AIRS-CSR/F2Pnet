# Semantic translation of face image with limited pixels for simulated prosthetic vision
In order to improve the intelligibility of face images with limited pixels, we construct a face semantic information transformation model to transform real faces into pixel faces based on the analogy between human intelligence and artificial intelligence, named F2Pnet (Face to Pixel networks).

Paper URL is coming soon

<img src="00.PNG">

## Requirements

- [Windows or Linux]
- [Tensorflow (2.5)]
- [tensorflow-addons]
- [OpenCV]

## Training strategy

The core idea of the training strategy is to transform the spatial semantic into more easy to learn masks and the spectral semantic into more easy to learn labels.

<img src="01.PNG">

## Dataset Preparation

    ├── main.py
    ├── checkpoint (Pretrained F2Pnet)
    ├── unet.h5 (Pretrained U-net)
    ├── AIRS-PFD (Our pixel face database) 
    	├── Pixel
			├── xxxxxx.png
			├── xxxxxx.png
			└── ...
		├── label.txt
    ├── CelebA
		├── celeba
			├── 000001.jpg 
			├── 000002.jpg
			└── ...
    	├── Anno
		    ├── list_attr_celeba_front.txt (We have provided) 
    ├── RafD-front (Please download RafD yourself for licensing reasons)
		├── xxxxxx.jpg 
		├── xxxxxx.jpg
		└── ...
    ├── test (The test image that you wanted)
        ├── a.jpg 
        ├── b.png
        └── ...

## Train

	python main.py --phase train

## Test

	## CelebA

	python main.py --phase test

	## test images

	python main.py --phase test --test_path ./test


## Pretrained model

- Download pretrained F2Pnet, pretrained U-net, and list_attr_celeba_front.txt here (https://pan.baidu.com/s/1i1LXF8ZrZRCtL2izEMelCQ) (uv6n)