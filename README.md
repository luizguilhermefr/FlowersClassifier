# Flowers Classifier

Classifies flowers using the [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset and MobileNetV2 through TensorFlow.

## Developing

First of all, create and activate a venv:

```sh
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```

Then, install the packages:

```sh
pip install -r requirements.txt
```

Once you are done developing, you can exit the virtual env:

```sh
deactivate
```

## Usage

Once you're in the virtual env, you can run:

```
python main.py <image location> <model location>
```

For example:

```
python main.py resources/cautleya_spicata.jpg model.h5
```

That it'll dump the most likely classes of flowers:

```
cautleya spicata...................99.38%
red ginger.........................0.46%
wallflower.........................0.05%
siam tulip.........................0.03%
monkshood..........................0.02%
```

You can also specify the following arguments:

- `--top_k` to set how many (most likely) classes of flowers you want, defaults to 5
