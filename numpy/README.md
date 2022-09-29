# Numpy
V této lekci si řekneme něco o knihovně [`numpy`](https://numpy.org/), která slouží pro práci s
n-dimenzionálními poli (tensory) a používá se jako "podvozek" pro naprostou většinu Python nástrojů
pro analýzu dat a strojové učení.

## Instalace závislosti
```bash
(venv) $ python3 -m pip install numpy imageio matplotlib
```

## Notebooky
- [Numpy](numpy.ipynb) - základy `numpy`, vytváření polí, indexace, filtrování, per-element operace s prvky,
broadcasting, redukce polí
- [Obrázky](images.ipynb) - použití `numpy` k manipulaci s obrázky

## Úlohy
Zadání úloh naleznete ve složce [tasks](tasks). K dispozici máte také unit testy, které můžete v této složce
spustit pomocí příkazu `$ python3 -m pytest tests.py`.

Potřebujete si k tomu nainstalovat knihovnu `pytest`:
```bash
(venv) $ python3 -m pip install pytest
```

## Materiály
- [Pyvec NumPy lekce](https://naucse.python.cz/course/mi-pyt/intro/numpy/)
- [Cheatsheet](https://pyvec.github.io/cheatsheets/numpy/numpy-cs.pdf)
- [Vizuální intro do NumPy](https://jalammar.github.io/visual-numpy/)
