netdissect
==========

[Network Dissection](http://netdissect.csail.mit.edu/) is a way
to inspect the internal representations of a deep convolutional
neural network to understand how internal units align with
human-interpretable concepts.

This package is a rewrite of netdissect to use native Pytorch
idioms: it depends on python 3 and pytorch 4.1.

## Setup

Depends on having the Broden dataset available.  Run
`script/dlbroden.sh` to download it into the `dataset/` directory.

Also, if `conda` is available, `script/setup_p3t41.sh` will create an
Anaconda environment with python 3 and a current build of pytorch 4.1.

## API

It can be used from code as a function, as follows:

1. Load up the convolutional model you wish to dissect, and call
   `retain_layers(model, [layernames,..])` to instrument the model.
2. Load the segmentation dataset using the BrodenDataset class;
   use the `transform_image` argument to normalize images to be
   suitable for the model, and the `size` argument to truncate the dataset.
3. Write a function to recover the original image (with RGB scaled to
   `[0...1]`) given a normalized dataset image; `ReverseNormalize` in this
   package inverts `torchvision.transforms.Normalize` for this purpose.
4. Choose a directory in which to write the output, and call
   `dissect(outdir, model, dataset)`.

A quick approximate dissection can be done by reducing the `size`
of the `BrodenDataset`.  Generating example images can be time-consuming
and the number of images can be set via `examples_per_unit`.

Example:

```
    from netdissect import retain_layers, dissect
    from netdissect import ReverseNormalize

    model = load_my_model()
    model.eval()
    model.cuda()
    retain_layers(model, ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
    bds = BrodenDataset('dataset/broden1_227',
            transform_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=10000)
    dissect('result/dissect', model, bds,
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            batch_size=100,
            examples_per_unit=10)
```

## Command Line

Net dissect command-line utility.  Usage:

```
python -m netdissect [-h] --model MODEL [--pthfile PTHFILE]
                     [--outdir OUTDIR] [--broden BRODEN]
                     [--layers LAYERS [LAYERS ...]] [--netname NETNAME]
                     [--imgsize IMGSIZE] [--examples EXAMPLES]
                     [--size SIZE] [--no-cuda] [--quiet]
```

optional arguments:

```
  -h, --help            show this help message and exit
  --model MODEL         constructor for the model to test
  --pthfile PTHFILE     filename of .pth file for the model
  --outdir OUTDIR       directory for dissection output
  --broden BRODEN       filename of Broden dataset
  --layers LAYERS [LAYERS ...]
                        list of layer names to dissect
  --netname NETNAME     name for network in generated reports
  --imgsize IMGSIZE     input image size to use
  --examples EXAMPLES   number of image examples per unit
  --size SIZE           dataset subset size to use
  --no-cuda             disables CUDA usage
  --quiet               silences console output
```

Example: to dissect three layers of the pretrained alexnet in torchvision:

```
python -m netdissect \
        --model "torchvision.models.alexnet(pretrained=True)" \
        --layers features.6:conv3 features.8:conv4 features.10:conv5 \
        --imgsize 227 \
        --outdir dissect/alexnet-imagenet
```
