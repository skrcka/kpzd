# Transfer learning
V této lekci si řekneme něco o tzv. `transfer learningu`, při kterém vezmeme již předtrénovanou síť
a tu poté pouze dotrénováváme, čímž můžeme výrazně urychlit trénovací proces.

Také si ukážeme augmentace a `tf.data`.

Dataset Cats vs Dogs lze najít na [Kaggle](https://www.kaggle.com/c/dogs-vs-cats). Stáhněte si jej
a rozbalte do složek `dogs-vs-cats/train` a `dogs-vs-cats/test1`.

## Instalace závislosti
```bash
$ pip install tensorflow tqdm pillow
```

## Soubory
- [Loader](loader.py) - loader pro dataset Cats vs Dogs.
- [Loader](loader_augmentation.py) - loader s augmentacemi.
- [Loader](loader_tfdata.py) - loader využívající `tf.data`.
- [Training](traing.py) - trénovací kód + definice modelu + transfer learning.

## Užitečné odkazy
- [Vizualizace konvoluce](https://ezyang.github.io/convolution-visualizer/)
- [CS231n - konvoluční sítě](https://cs231n.github.io/convolutional-networks/)
