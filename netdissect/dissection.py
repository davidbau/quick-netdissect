'''
To run dissection:

1. Load up the convolutional model you wish to dissect, and call
   retain_layers(model, [layernames,..]) to instrument the model.
2. Load the segmentation dataset using the BrodenDataset class;
   use the transform_image argument to normalize images to be
   suitable for the model, or the size argument to truncate the dataset.
3. Write a function to recover the original image (with RGB scaled to
   [0...1]) given a normalized dataset image; ReverseNormalize in this
   package inverts transforms.Normalize for this purpose.
4. Choose a directory in which to write the output, and call
   dissect(outdir, model, dataset).

Example:

    from dissect import retain_layers, dissect
    from broden import BrodenDataset

    model = load_my_model()
    model.eval()
    model.cuda()
    retain_layers(model, ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
    bds = BrodenDataset('dataset/broden1_227',
            transform_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=1000)
    dissect('result/dissect', model, bds,
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            examples_per_unit=10)
'''

import torch, numpy, os, re, json, shutil
from PIL import Image
from xml.etree import ElementTree as et
from collections import OrderedDict, defaultdict
from .progress import verbose_progress, default_progress, print_progress
from .runningstats import RunningQuantile, RunningTopK
from .sampler import FixedSubsetSampler
from .actviz import activation_visualization
from .segviz import segment_visualization

def dissect(outdir, model, dataset,
        recover_image=None,
        quantile_threshold=0.005,
        iou_threshold=0.04,
        examples_per_unit=100,
        batch_size=100,
        num_workers=24,
        make_images=True,
        make_labels=True,
        make_report=True,
        make_single_images=False,
        netname=None,
        meta=None,
        settings=None,
        ):
    '''
    Runs net dissection in-memory, using pytorch, and saves visualizations
    and metadata into outdir.
    '''
    assert not model.training, 'Run model.eval() before dissection'
    if netname is None:
        netname = type(model).__name__
    with torch.no_grad():
        device = next(model.parameters()).device
        segloader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size, num_workers=num_workers,
                pin_memory=(device.type == 'cuda'))
        quantiles, topk = collect_quantiles_and_topk(model, segloader,
                recover_image=recover_image, k=examples_per_unit)
        levels = {k: qc.quantiles([1.0 - quantile_threshold])[:,0]
                for k, qc in quantiles.items()}
        quantiledata = (topk, quantiles, levels, quantile_threshold)
        if make_images:
            generate_images(outdir, model, dataset, topk, levels, recover_image,
                    row_length=examples_per_unit, batch_size=batch_size,
                    single_images=make_single_images,
                    num_workers=num_workers)
        if make_labels:
            if hasattr(recover_image, 'get_label_and_category_names'):
                labelnames, catnames = (
                        recover_image.get_label_and_category_names(dataset))
            else:
                labelnames, catnames = broden_label_and_category_names(dataset)
            label_category = [catnames.index(c) for l, c in labelnames]
            segloader = torch.utils.data.DataLoader(dataset,
                    batch_size=1, num_workers=num_workers,
                    pin_memory=(device.type == 'cuda'))
            lcs, ccs, ics = collect_bincounts(model, segloader, levels,
                    recover_image=recover_image)
            scores = {
                    k: score_tally_stats(label_category, lcs, ccs[k], ics[k])
                    for k in ics}
            labeldata = (labelnames, catnames, scores, lcs, ccs, ics,
                    iou_threshold)
        else:
            labeldata = None
        if make_report:
            generate_report(outdir,
                    quantiledata=quantiledata,
                    labeldata=labeldata,
                    netname=netname,
                    meta=meta,
                    settings=settings)
        return quantiledata, labeldata

def generate_report(outdir, quantiledata, labeldata=None,
        netname='Model', meta=None, settings=None):
    '''
    Creates dissection.json reports and summary bargraph.svg files in the
    specified output directory, and copies a dissection.html interface
    to go along with it.
    '''
    all_layers = []
    # Current source code directory, for html to copy.
    srcdir = os.path.realpath(
       os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # Unpack arguments
    topk, quantiles, levels, quantile_threshold = quantiledata
    top_record = dict(
            netname=netname,
            meta=meta,
            default_ranking='unit',
            quantile_threshold=quantile_threshold)
    if settings is not None:
        top_record['settings'] = settings
    if labeldata is not None:
        labelnames, catnames, scores, lcs, ccs, ics, iou_threshold = labeldata
        catorder = {'object': -6, 'scene': -5, 'part': -4,
                    'material': -3, 'texture': -2, 'color': -1}
        for i, cat in enumerate(c for c in catnames if c not in catorder):
            catorder[cat] = i
        catnumber = {n: i for i, n in enumerate(catnames)}
        top_record['default_ranking'] = 'label'
        top_record['iou_threshold'] = iou_threshold
    for layer in topk.keys():
        units, rankings = [], []
        record = dict(layer=layer, units=units, rankings=rankings)
        # For every unit, we always have basic visualization information.
        topa, topi = topk[layer].result()
        lev = levels[layer]
        for u in range(len(topa)):
            units.append(dict(
                unit=u,
                interp=True,
                level=lev[u].item(),
                top=[dict(imgnum=i.item(), maxact=a.item())
                    for i, a in zip(topi[u], topa[u])],
                ))
        rankings.append(dict(name="unit", score=list([
            u for u in range(len(topa))])))
        # TODO: consider including stats and ranking based on quantiles,
        # variance, connectedness here.

        # if we have labeldata, then every unit also gets a bunch of other info
        if labeldata is not None:
            labelunits = defaultdict(list)
            lscore, cc, ic = [dat[layer] for dat in [scores, ccs, ics]]
            best_score, best_label = lscore.max(0)
            record['iou_threshold'] = iou_threshold,
            for u, urec in enumerate(units):
                score, label = best_score[u], best_label[u]
                urec.update(dict(
                    iou=score.item(),
                    lc=lcs[label].item(),
                    cc=cc[catnumber[labelnames[label][1]], u].item(),
                    ic=ic[label, u].item(),
                    interp=(score.item() > iou_threshold),
                    labelnum=label.item(),
                    label=labelnames[label.item()][0],
                    cat=labelnames[label.item()][1],
                    ))
                if score.item() > iou_threshold:
                    labelunits[label.item()].append(u)
            # Sort all units in order with most popular label first.
            label_ordering = sorted(units,
                # Sort by:
                key=lambda r: (-1 if r['interp'] else 0,  # interpretable
                    -len(labelunits[r['labelnum']]),      # label freq, score
                    -max([units[u]['iou'] for u in labelunits[r['labelnum']]],
                        default=0),
                    r['labelnum'],                        # label
                    -r['iou']))                           # unit score
            # Add label and iou ranking.
            rankings.append(dict(name="label", score=numpy.argsort(list(
                ur['unit'] for ur in label_ordering)).tolist()))
            rankings.append(dict(name="iou", score=list(
                -ur['iou'] for ur in units)))
            # Add ranking for top labels
            for labelnum in [n for n in sorted(labelunits.keys(), key=lambda x:
                    -len(labelunits[x])) if len(labelunits[n])]:
                label = labelnames[labelnum][0]
                rankings.append(dict(name="%s-iou" % label,
                    concept=label, metric='iou',
                    score=(-lscore[labelnum, :]).tolist()))
            # Add argsort to every ranking
            for r in rankings:
                r['ranking'] = numpy.argsort(r['score']).tolist()
            # Collate labels by category then frequency.
            record['labels'] = [dict(
                        label=labelnames[label][0],
                        labelnum=label,
                        units=labelunits[label],
                        cat=labelnames[label][1])
                    for label in (sorted(labelunits.keys(),
                        # Sort by:
                        key=lambda l: (catorder.get(          # category
                            labelnames[l][1], 0),
                            -len(labelunits[l]),              # label freq
                            -max([units[u]['iou'] for u in labelunits[l]],
                                default=0) # score
                            ))) if len(labelunits[label])]
            # Total number of interpretable units.
            record['interpretable'] = sum(len(group['units'])
                    for group in record['labels'])
            # Make a bargraph of labels
            os.makedirs(os.path.join(outdir, safe_dir_name(layer)),
                    exist_ok=True)
            catgroups = OrderedDict()
            for _, cat in sorted([(v, k) for k, v in catorder.items()]):
                catgroups[cat] = []
            for rec in record['labels']:
                if rec['cat'] not in catgroups:
                    catgroups[rec['cat']] = []
                catgroups[rec['cat']].append(rec['label'])
            make_svg_bargraph(
                    [rec['label'] for rec in record['labels']],
                    [len(rec['units']) for rec in record['labels']],
                    [(cat, len(group)) for cat, group in catgroups.items()],
                    filename=os.path.join(outdir, safe_dir_name(layer),
                        'bargraph.svg'))
            # Only show the bargraph if it is non-empty.
            if len(record['labels']):
                record['bargraph'] = 'bargraph.svg'
        # Dump per-layer json inside per-layer directory
        record['dirname'] = '.'
        with open(os.path.join(outdir, safe_dir_name(layer), 'dissect.json'),
                'w') as jsonfile:
            top_record['layers'] = [record]
            json.dump(top_record, jsonfile, indent=1)
        # Copy the per-layer html
        shutil.copy(os.path.join(srcdir, 'dissect.html'),
                os.path.join(outdir, safe_dir_name(layer), 'dissect.html'))
        record['dirname'] = safe_dir_name(layer)
        all_layers.append(record)
    # Dump all-layer json in parent directory
    with open(os.path.join(outdir, 'dissect.json'), 'w') as jsonfile:
        top_record['layers'] = all_layers
        json.dump(top_record, jsonfile, indent=1)
    # Copy the all-layer html
    shutil.copy(os.path.join(srcdir, 'dissect.html'),
            os.path.join(outdir, 'dissect.html'))


def generate_images(outdir, model, dataset, topk, levels,
        recover_image=None, row_length=None, gap_pixels=5,
        single_images=False, batch_size=100, num_workers=24):
    '''
    Creates an image strip file for every unit of every retained layer
    of the model, in the format [outdir]/[layername]/[unitnum]-top.jpg.
    Assumes that the indexes of topk refer to the indexes of dataset.
    To recover RGB images from a normalized dataset, pass a reverse
    normalization function as recover_image.
    Limits each strip to the top row_length images.
    '''
    progress = default_progress()
    needed_images = {}
    if recover_image is None:
        recover_image = lambda x: x
    # Pass 1: needed_images lists all images that are topk for some unit.
    for layer in topk:
        topresult = topk[layer].result()[1].cpu()
        for unit, row in enumerate(topresult):
            for rank, imgnum in enumerate(row[:row_length]):
                imgnum = imgnum.item()
                if imgnum not in needed_images:
                    needed_images[imgnum] = []
                needed_images[imgnum].append((layer, unit, rank))
    levels = {k: v.cpu().numpy() for k, v in levels.items()}
    row_length = len(row[:row_length])
    needed_sample = FixedSubsetSampler(sorted(needed_images.keys()))
    device = next(model.parameters()).device
    segloader = torch.utils.data.DataLoader(dataset,
            batch_size=5, num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            sampler=needed_sample)
    vizgrid, maskgrid, origrid, seggrid = [{} for _ in range(4)]
    # Pass 2: populate vizgrid with visualizations of top units.
    for i, batch in enumerate(
            progress(segloader, desc='Making images')):
        # Reverse transformation to get the image in byte form.
        # if hasattr(recover_image, 'recover_image_and_features'):
        #     byte_im, features, scale_offset = (
        #             recover_image.recover_image_and_features(
        #                batch, model))
        if hasattr(recover_image, 'recover_im_seg_bc_and_features'):
            byte_im, seg, _, features, scale_offset = (
                    recover_image.recover_im_seg_bc_and_features(
                        batch, model, byte_images=True))
        else:
            im, seg, bc = batch
            byte_im = recover_image(im.clone()
                    ).permute(0, 2, 3, 1).mul_(255).clamp(0, 255).byte()
            # Run the model.
            model(im.to(device))
            features = model.retained
            scale_offset = getattr(model, 'scale_offset', None)
        byte_im = byte_im.cpu().numpy()
        features = {k: v.cpu().numpy() for k, v in features.items()}
        for index in range(len(byte_im)):
            imgnum = needed_sample.samples[index + i*segloader.batch_size]
            for layer, unit, rank in needed_images[imgnum]:
                acts = features[layer]
                if layer not in vizgrid:
                    [vizgrid[layer], maskgrid[layer], origrid[layer],
                            seggrid[layer]] = [
                        numpy.full((acts.shape[1], byte_im.shape[1], row_length,
                            byte_im.shape[2] + gap_pixels, depth), 255,
                            dtype='uint8')
                        for depth in [3, 4, 3, 3]]
                origrid[layer][unit,:,rank,:byte_im.shape[1],:] = byte_im[index]
                [vizgrid[layer][unit,:,rank,:byte_im.shape[1],:],
                 maskgrid[layer][unit,:,rank,:byte_im.shape[1],:]] = (
                    activation_visualization(
                        byte_im[index],
                        acts[index, unit],
                        levels[layer][unit],
                        scale_offset=scale_offset[layer]
                                     if scale_offset else None,
                        return_mask=True))
                seggrid[layer][unit,:,rank,:byte_im.shape[1],:] = (
                    segment_visualization(seg[index].cpu().numpy(),
                        byte_im.shape[1:3]))
    # Pass 3: save image strips as [outdir]/[layer]/[unitnum]-[top/orig].jpg
    for layer, vg in progress(vizgrid.items(), desc='Saving images'):
        os.makedirs(os.path.join(outdir, safe_dir_name(layer), 'image'),
                exist_ok=True)
        og, sg, mg = origrid[layer], seggrid[layer], maskgrid[layer]
        for unit in progress(range(len(vg)), desc='Units'):
            for suffix, grid in [('top.jpg', vg), ('orig.jpg', og),
                    ('seg.png', sg), ('mask.png', mg)]:
                strip = grid[unit].reshape(
                        (grid.shape[1], grid.shape[2] * grid.shape[3],
                            grid.shape[4]))
                filename = os.path.join(outdir, safe_dir_name(layer),
                        'image', '%d-%s' % (unit, suffix))
                Image.fromarray(strip[:,:-gap_pixels,:]).save(filename,
                        optimize=True, quality=80)
                if single_images:
                    single_filename = os.path.join(outdir, safe_dir_name(layer),
                        'image', 's-%d-%s' % (unit, suffix))
                    Image.fromarray(strip[:,:strip.shape[1] // row_length
                        - gap_pixels,:]).save(single_filename,
                                optimize=True, quality=80)

def score_tally_stats(label_category, lc, cc, ic):
    ec = cc[label_category]
    epsilon = 1e-20 # avoid division-by-zero
    iou = ic.double() / ((ec + lc[:,None] - ic).double() + epsilon)
    return iou

def collect_quantiles_and_topk(model, segloader,
        recover_image=None, k=100, resolution=1024):
    '''
    Collects (estimated) quantile information and (exact) sorted top-K lists
    for every channel in the retained layers of the model.  Returns
    a map of quantiles (one RunningQuantile for each layer) along with
    a map of topk (one RunningTopK for each layer).
    '''
    quantiles = {}
    topks = {}
    device = next(model.parameters()).device
    progress = default_progress()
    for i, batch in enumerate(progress(segloader, desc='Quantiles')):
        # We don't actually care about the model output.
        model(batch[0].to(device))
        features = model.retained
        # We care about the retained values
        for key, value in features.items():
            if key not in topks:
                topks[key] = RunningTopK(k)
            if key not in quantiles:
                quantiles[key] = RunningQuantile(
                        depth=value.shape[1], resolution=resolution,
                        dtype=value.dtype, device=device)
            topvalue = value
            if len(value.shape) > 2:
                topvalue, _ = value.view(*(value.shape[:2] + (-1,))).max(2)
                # Put the channel index last.
                value = value.permute(
                        (0,) + tuple(range(2, len(value.shape))) + (1,)
                        ).contiguous().view(-1, value.shape[1])
            quantiles[key].add(value)
            topks[key].add(topvalue)
    return quantiles, topks

def collect_bincounts(model, segloader, levels, recover_image=None):
    '''
    Returns label_counts, category_activation_counts, and intersection_counts,
    across the data set, counting the pixels of intersection between upsampled,
    thresholded model featuremaps, with segmentation classes in the segloader.

    label_counts (independent of model): pixels across the data set that
        are labeled with the given label.
    category_activation_counts (one per layer): for each feature channel,
        pixels across the dataset where the channel exceeds the level
        threshold.  There is one count per category: activations only
        contribute to the categories for which any category labels are
        present on the images.
    intersection_counts (one per layer): for each feature channel and
        label, pixels across the dataset where the channel exceeds
        the level, and the labeled segmentation class is also present.

    This is a performance-sensitive function.  Best performance is
    achieved with a counting scheme which assumes a segloader with
    batch_size 1.
    '''
    device = next(model.parameters()).device
    if hasattr(recover_image, 'get_label_and_category_names'):
        labelcat, categories = recover_image.get_label_and_category_names(
                segloader.dataset)
        label_category = [categories.index(c) for l, c in labelcat]
        num_labels, num_categories = (len(n) for n in [labelcat, categories])
    else:
        num_labels = len(segloader.dataset.labels)
        num_categories = len(segloader.dataset.categories)
        label_category = segloader.dataset.label_category
    # One-hot vector of category for each label
    labelcat = torch.zeros(num_labels, num_categories,
            dtype=torch.long, device=device)
    labelcat.scatter_(1, torch.from_numpy(numpy.array(label_category,
        dtype='int64')).to(device)[:,None], 1)
    # Running bincounts
    # activation_counts = {}
    assert segloader.batch_size == 1 # category_activation_counts needs this.
    category_activation_counts = {}
    intersection_counts = {}
    label_counts = torch.zeros(num_labels, dtype=torch.long, device=device)
    progress = default_progress()
    scale_offset_map = getattr(model, 'scale_offset', None)
    upsample_grids = {}
    # total_batch_categories = torch.zeros(
    #         labelcat.shape[1], dtype=torch.long, device=device)
    for i, batch in enumerate(progress(segloader, desc='Bincounts')):
        if hasattr(recover_image, 'recover_im_seg_bc_and_features'):
            im, seg, batch_label_counts, features, scale_offset_map = (
                    recover_image.recover_im_seg_bc_and_features(
                        batch, model))
            bc = batch_label_counts.cpu()
        else:
            bc = batch[2]
            im, seg, batch_label_counts = (d.to(device) for d in batch)
            model(im)
            features = model.retained
        # Accumulate bincounts and identify nonzeros
        label_counts += batch_label_counts[0]
        batch_labels = bc[0].nonzero()[:,0]
        batch_categories = labelcat[batch_labels].max(0)[0]
        for key, value in features.items():
            if key not in upsample_grids:
                upsample_grids[key] = upsample_grid(value.shape[2:],
                        seg.shape[2:], im.shape[2:],
                        scale_offset=scale_offset_map.get(key, None)
                            if scale_offset_map is not None else None,
                        dtype=value.dtype, device=value.device)
            upsampled = torch.nn.functional.grid_sample(value,
                    upsample_grids[key], padding_mode='border')
            amask = (upsampled > levels[key][None,:,None,None])
            ac = amask.int().view(amask.shape[1], -1).sum(1)
            # if key not in activation_counts:
            #     activation_counts[key] = ac
            # else:
            #     activation_counts[key] += ac
            # The fastest approach: sum over each label separately!
            for label in batch_labels.tolist():
                imask = amask * ((seg == label).max(dim=1, keepdim=True)[0])
                ic = imask.int().view(imask.shape[1], -1).sum(1)
                if key not in intersection_counts:
                    intersection_counts[key] = torch.zeros(num_labels,
                            amask.shape[1], dtype=torch.long, device=device)
                intersection_counts[key][label] += ic
            # Count activations within images that have category labels.
            # Note: This only makes sense with batch-size one
            # total_batch_categories += batch_categories
            cc = batch_categories[:,None] * ac[None,:]
            if key not in category_activation_counts:
                category_activation_counts[key] = cc
            else:
                category_activation_counts[key] += cc
    return (label_counts, category_activation_counts, intersection_counts)

def upsample_grid(data_shape, target_shape, input_shape=None,
        scale_offset=None, dtype=torch.float, device=None):
    '''Prepares a grid to use with grid_sample to upsample a batch of
    features in data_shape to the target_shape. Can use scale_offset
    and input_shape to center the grid in a nondefault way: scale_offset
    maps feature pixels to input_shape pixels, and it is assumed that
    the target_shape is a uniform downsampling of input_shape.'''
    # Default is that nothing is resized.
    if target_shape is None:
        target_shape = data_shape
    # Make a default scale_offset to fill the image if there isn't one
    if scale_offset is None:
        scale = tuple(float(ts) / ds
                for ts, ds in zip(target_shape, data_shape))
        offset = tuple(0.5 * s - 0.5 for s in scale)
    else:
        scale, offset = (v for v in zip(*scale_offset))
        # Handle downsampling for different input vs target shape.
        if input_shape is not None:
            scale = tuple(s * (ts - 1) / (ns - 1)
                    for s, ns, ts in zip(scale, input_shape, target_shape))
            offset = tuple(o * (ts - 1) / (ns - 1)
                    for o, ns, ts in zip(offset, input_shape, target_shape))
    # Pytorch needs target coordinates in terms of source coordinates [-1..1]
    ty, tx = (((torch.arange(ts, dtype=dtype, device=device) - o)
                  * (2 / (s * (ss - 1))) - 1)
        for ts, ss, s, o, in zip(target_shape, data_shape, scale, offset))
    # Whoa, note that grid_sample reverses the order y, x -> x, y.
    grid = torch.stack(
        (tx[None,:].expand(target_shape), ty[:,None].expand(target_shape)),2
       )[None,:,:,:].expand((1, target_shape[0], target_shape[1], 2))
    return grid

def dilation_scale_offset(dilations):
    '''Composes a list of (k, s, p) into a single total scale and offset.'''
    if len(dilations) == 0:
        return (1, 0)
    scale, offset = dilation_scale_offset(dilations[1:])
    kernel, stride, padding = dilations[0]
    scale *= stride
    offset *= stride
    offset += (kernel - 1) / 2.0 - padding
    return scale, offset

def dilations(modulelist):
    '''Converts a list of modules to (kernel_size, stride, padding)'''
    result = []
    for module in modulelist:
        settings = tuple(getattr(module, n, d)
            for n, d in (('kernel_size', 1), ('stride', 1), ('padding', 0)))
        settings = (((s, s) if not isinstance(s, tuple) else s)
            for s in settings)
        if settings != ((1, 1), (1, 1), (0, 0)):
            result.append(zip(*settings))
    return zip(*result)

def sequence_scale_offset(modulelist):
    '''Returns (yscale, yoffset), (xscale, xoffset) given a list of modules'''
    return tuple(dilation_scale_offset(d) for d in dilations(modulelist))

def retain_layer_output(dest, layer, name):
    '''Callback function to keep a reference to a layer's output in a dict.'''
    def hook_fn(m, i, output):
        dest[name] = output.detach()
    layer.register_forward_hook(hook_fn)

def retain_layers(model, layer_names, add_scale_offset=True):
    '''
    Creates a 'retained' property on the model which will keep a record
    of the layer outputs for the specified layers.  Also computes the
    cumulative scale and offset for convolutions.

    The layer_names array should be a list of layer names, or tuples
    of (name, aka) where the name is the pytorch name for the layer,
    and the aka string is the name you wish to use for the dissection.
    '''
    model.retained = {}
    if add_scale_offset:
        model.scale_offset = {}
    seen = set()
    sequence = []
    aka_map = {}
    for name in layer_names:
        aka = name
        if not isinstance(aka, str):
            name, aka = name
        aka_map[name] = aka
    for name, layer in model.named_modules():
        sequence.append(layer)
        if name in aka_map:
            seen.add(name)
            aka = aka_map[name]
            retain_layer_output(model.retained, layer, aka)
            if add_scale_offset:
                model.scale_offset[aka] = sequence_scale_offset(sequence)
    for name in aka_map:
        assert name in seen, ('Layer %s not found' % name)

def broden_label_and_category_names(dataset):
    catnames = dataset.categories
    label_and_cat_names = [(readable(label),
        catnames[dataset.label_category[i]])
            for i, label in enumerate(dataset.labels)]
    return label_and_cat_names, catnames

def safe_dir_name(filename):
    keepcharacters = (' ','.','_','-')
    return ''.join(c
            for c in filename if c.isalnum() or c in keepcharacters).rstrip()

bargraph_palette = [
    ('#4B4CBF', '#B6B6F2'),
    ('#55B05B', '#B6F2BA'),
    ('#50BDAC', '#A5E5DB'),
    ('#D4CF24', '#F2F1B6'),
    ('#F0883B', '#F2CFB6'),
    ('#D92E2B', '#F2B6B6')
]

def make_svg_bargraph(labels, heights, categories,
        barheight=100, barwidth=12, show_labels=True, filename=None):
    # if len(labels) == 0:
    #     return # Nothing to do
    unitheight = float(barheight) / max(max(heights, default=1), 1)
    textheight = barheight if show_labels else 0
    labelsize = float(barwidth)
    gap = float(barwidth) / 4
    textsize = barwidth + gap
    rollup = max(heights, default=1)
    textmargin = float(labelsize) * 2 / 3
    leftmargin = 32
    rightmargin = 8
    svgwidth = len(heights) * (barwidth + gap) + 2 * leftmargin + rightmargin
    svgheight = barheight + textheight

    # create an SVG XML element
    svg = et.Element('svg', width=str(svgwidth), height=str(svgheight),
            version='1.1', xmlns='http://www.w3.org/2000/svg')
 
    # Draw the bar graph
    basey = svgheight - textheight
    x = leftmargin
    # Add units scale on left
    if len(heights):
        for h in [1, (max(heights) + 1) // 2, max(heights)]:
            et.SubElement(svg, 'text', x='0', y='0',
                style=('font-family:sans-serif;font-size:%dpx;' +
                'text-anchor:end;alignment-baseline:hanging;' +
                'transform:translate(%dpx, %dpx);') %
                (textsize, x - gap, basey - h * unitheight)).text = str(h)
        et.SubElement(svg, 'text', x='0', y='0',
                style=('font-family:sans-serif;font-size:%dpx;' +
                'text-anchor:middle;' +
                'transform:translate(%dpx, %dpx) rotate(-90deg)') %
                (textsize, x - gap - textsize, basey - h * unitheight / 2)
                ).text = 'units'
    # Draw big category background rectangles
    for catindex, (cat, catcount) in enumerate(categories):
        if not catcount:
            continue
        et.SubElement(svg, 'rect', x=str(x), y=str(basey - rollup * unitheight),
                width=(str((barwidth + gap) * catcount - gap)),
                height = str(rollup*unitheight),
                fill=bargraph_palette[catindex % len(bargraph_palette)][1])
        x += (barwidth + gap) * catcount
    # Draw small bars as well as 45degree text labels
    x = leftmargin
    catindex = -1
    catcount = 0
    for label, height in zip(labels, heights):
        while not catcount and catindex <= len(categories):
            catindex += 1
            catcount = categories[catindex][1]
            color = bargraph_palette[catindex % len(bargraph_palette)][0]
        et.SubElement(svg, 'rect', x=str(x), y=str(basey-(height * unitheight)),
                width=str(barwidth), height=str(height * unitheight),
                fill=color)
        x += barwidth
        if show_labels:
            et.SubElement(svg, 'text', x='0', y='0',
                style=('font-family:sans-serif;font-size:%dpx;text-anchor:end;'+
                'transform:translate(%dpx, %dpx) rotate(-45deg);') %
                (labelsize, x, basey + textmargin)).text = readable(label)
        x += gap
        catcount -= 1
    # Text labels for each category
    x = leftmargin
    for cat, catcount in categories:
        if not catcount:
            continue
        et.SubElement(svg, 'text', x='0', y='0',
            style=('font-family:sans-serif;font-size:%dpx;text-anchor:end;'+
            'transform:translate(%dpx, %dpx) rotate(-90deg);') %
            (textsize, x + (barwidth + gap) * catcount - gap,
                basey - rollup * unitheight + gap)).text = '%d %s' % (
                    catcount, readable(cat + ('s' if catcount != 1 else '')))
        x += (barwidth + gap) * catcount
    # Output - this is the bare svg.
    result = et.tostring(svg)
    if filename:
        f = open(filename, 'wb')
        # When writing to a file a special header is needed.
        f.write(''.join([
            '<?xml version=\"1.0\" standalone=\"no\"?>\n',
            '<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n',
            '\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n']
            ).encode('utf-8'))
        f.write(result)
        f.close()
    return result

readable_replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]

def readable(label):
    for pattern, subst in readable_replacements:
        label= re.sub(pattern, subst, label)
    return label

class ReverseNormalize:
    def __init__(self, mean, stdev):
        mean = numpy.array(mean)
        stdev = numpy.array(stdev)
        self.mean = torch.from_numpy(mean)[None,:,None,None].float()
        self.stdev = torch.from_numpy(stdev)[None,:,None,None].float()
    def __call__(self, data):
        return data.mul_(self.stdev).add_(self.mean)

