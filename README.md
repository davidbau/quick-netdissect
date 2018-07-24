quick-netdissect
================

This is a rewrite of netdissect to use native Pytorch idioms.
It can be used from code as a function, as follows:

Depends on python 3 and pytorch 4.1.

To run dissection:

1. Load up the convolutional model you wish to dissect, and call
   `retain_layers(model, [layernames,..])` to instrument the model.
2. Load the segmentation dataset using the BrodenDataset class;
   use the `transform_image` argument to normalize images to be
   suitable for the model, or the size argument to truncate the dataset.
3. Write a function to recover the original image (with RGB scaled to
   `[0...1]`) given a normalized dataset image; `ReverseNormalize` in this
   package inverts `torchvision.transforms.Normalize` for this purpose.
4. Choose a directory in which to write the output, and call
   `dissect(outdir, model, dataset)`.

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

Dissection can also be run from the command-line using `python -m netdissect`.

For example, to dissect three layers of the pretrained alexnet in torchvision:

```
python -m netdissect \
        --model "torchvision.models.alexnet(pretrained=True)" \
        --layers features.6:conv3 features.8:conv4 features.10:conv5 \
        --imgsize 227 \
        --outdir dissect/alexnet-imagenet
```
