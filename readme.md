# DeepSomatic

Classify somatic/germline mutations by deep neural networks.

There is also [a version for Julia](https://github.com/OpenGene/DeepSomatic/tree/master).

### Installation

1. Python 3.5+
2. Keras with TensorFlow by their [guide](https://keras.io/#installation)
3. `pip install pysam`

### predict

```
$ python predict.py ../model/model2.h5 input.bam input.txt > output.txt
```

By default it accept the `txt` file generated by AnnoVar.

### train new models

1. run `python collect_data.py cfDNA.bam gDNA.mpileup out.image out.txt` for each sample to get a list of `.image` and `.txt` files.
2. run `python model1.py` to train the corresponding model.