import torch, sys, os, argparse, textwrap, numbers, numpy, json
from torchvision import transforms
from netdissect.progress import verbose_progress, print_progress
from netdissect import retain_layers, BrodenDataset, dissect, ReverseNormalize

help_epilog = '''\
Example: to dissect three layers of the pretrained alexnet in torchvision:

python -m netdissect \\
        --model "torchvision.models.alexnet(pretrained=True)" \\
        --layers features.6:conv3 features.8:conv4 features.10:conv5 \\
        --imgsize 227 \\
        --outdir dissect/alexnet-imagenet
'''

def main():
    # Training settings
    def strpair(arg):
        p = tuple(arg.split(':'))
        if len(p) == 1:
            p = p + p
        return p
    def intpair(arg):
        p = arg.split(',')
        if len(p) == 1:
            p = p + p
        return tuple(int(v) for v in p)

    parser = argparse.ArgumentParser(description='Net dissect utility',
            prog='python -m netdissect',
            epilog=textwrap.dedent(help_epilog),
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None, required=True,
                        help='constructor for the model to test')
    parser.add_argument('--pthfile', type=str, default=None,
                        help='filename of .pth file for the model')
    parser.add_argument('--outdir', type=str, default='dissect',
                        help='directory for dissection output')
    parser.add_argument('--broden', type=str, default=None,
                        help='filename of Broden dataset')
    parser.add_argument('--layers', type=strpair, nargs='+',
                        help='space-separated list of layer names to dissect')
    parser.add_argument('--meta', type=str, nargs='+',
                        help='filenames of metadata json files')
    parser.add_argument('--netname', type=str, default=None,
                        help='name for network in generated reports')
    parser.add_argument('--imgsize', type=intpair, default=(227, 227),
                        help='input image size to use')
    parser.add_argument('--examples', type=int, default=20,
                        help='number of image examples per unit')
    parser.add_argument('--size', type=int, default=10000,
                        help='dataset subset size to use')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for forward pass')
    parser.add_argument('--num_workers', type=int, default=24,
                        help='number of DataLoader workers')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA usage')
    parser.add_argument('--add_scale_offset', action='store_true', default=None,
                        help='offsets masks according to stride and padding')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='silences console output')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # default='torchvision.models.alexnet(pretrained=True)'

    # Set up console output
    verbose_progress(not args.quiet)

    # Speed up pytorch
    torch.backends.cudnn.benchmark = True

    # Construct the network
    if args.model is None:
        print_progress('No model specified')
        sys.exit(1)
    model = eval_constructor(args.model)
    # Default add_scale_offset only for AlexNet-looking models.
    if args.add_scale_offset is None:
        args.add_scale_offset = ('Alex' in model.__class__.__name__)

    # Load its state dict
    meta = {}
    if args.pthfile is None:
        print_progress('Dissecting model without pth file.')
    else:
        data = torch.load(args.pthfile)
        if 'state_dict' in data:
            meta = {}
            for key in data:
                if isinstance(data[key], numbers.Number):
                    meta[key] = data[key]
            data = data['state_dict']
        model.load_state_dict(data)

    # Update any metadata from files, if any
    for mfilename in args.meta:
        with open(mfilename) as f:
            meta.update(json.load(f))

    # Instrument it and prepare it for eval
    if not args.layers:
        print_progress('No layers specified')
        sys.exit(1)
    retain_layers(model, args.layers, args.add_scale_offset)
    model.eval()
    if args.cuda:
        model.cuda()

    # Set up the output directory, verify write access
    if args.outdir is None:
        args.outdir = os.path.join('dissect', type(model).__name__)
        print_progress('Writing output into %s.' % args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    # Load broden dataset
    brodendir = args.broden
    if brodendir is None:
        ds_resolution = (224 if max(args.imgsize) <= 224 else
                         227 if max(args.imgsize) <= 227 else 384)
        brodendir = os.path.join('dataset', 'broden1_%d' % ds_resolution)
        # TODO: autodownload perhaps.
        print_progress('Loading broden from %s' % brodendir)
    bds = BrodenDataset(brodendir,
            transform_image=transforms.Compose([
                transforms.Resize(args.imgsize),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=args.size)

    # Run dissect
    dissect(args.outdir, model, bds,
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            examples_per_unit=args.examples,
            netname=args.netname,
            meta=meta,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

def test_dissection():
    verbose_progress(True)
    from torchvision.models import alexnet
    from torchvision import transforms
    model = alexnet(pretrained=True)
    model.eval()
    # Load an alexnet
    retain_layers(model, [
        ('features.0', 'conv1'),
        ('features.3', 'conv2'),
        ('features.6', 'conv3'),
        ('features.8', 'conv4'),
        ('features.10', 'conv5') ])
    # load broden dataset
    bds = BrodenDataset('dataset/broden1_227',
            transform_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=100)
    # run dissect
    dissect('dissect/test', model, bds,
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            examples_per_unit=10)

def eval_constructor(term, construct_types=True):
    '''
    Used to evaluate an arbitrary command-line constructor specifying
    a class, with automatic import of global module names.
    '''
    from collections import defaultdict
    from importlib import import_module

    class DictNamespace(object):
        def __init__(self, d):
            self.__d__ = d
        def __getattr__(self, key):
            return self.__d__[key]

    class AutoImportDict(defaultdict):
        def __init__(self, parent=None):
            super().__init__()
            self.parent = parent
        def __missing__(self, key):
            if self.parent is not None:
                key = self.parent + '.' + key
            if hasattr(__builtins__, key):
                return getattr(__builtins__, key)
            mdl = import_module(key)
            # Return an AutoImportDict for any namespace packages
            if hasattr(mdl, '__path__') and not hasattr(mdl, '__file__'):
                return DictNamespace(AutoImportDict(key))
            return mdl

    obj = eval(term, {}, AutoImportDict())
    if isinstance(obj, type):
        obj = obj()
    return obj

# Many models use this normalization.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STDEV = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    main()
