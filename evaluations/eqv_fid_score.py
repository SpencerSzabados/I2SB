
"""

python eqv_fid_score.py --path "/home/datasets/c4_mnist_50000/" "/home/SP-GAN/fid/fid_samples/" --batch_size 128 --device cuda:0 --img_size 28 --eqv C4
"""


# from fid_score import *
from evaluations.fid_score import *

## Global arguments
#------------------------------------------------
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--img_size', type=int, default=28,
                    help="Size to crop images to while computing fid. Should be set to the full image resulution when possible.")
parser.add_argument('--save-stats', action='store_true',
                    help=('Generate an npz archive from a directory of samples. '
                          'The first path is used as input and the second as output.'))
parser.add_argument('--path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--eqv', type=str, default='Z2',
                    help="Equivariant group prior.")

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp', "JPEG"}



def calculate_eqv_diff_fid_given_paths(paths, batch_size, device, dims, img_size, num_workers=1, eqv='Z2'):
    """Calculates the FID of two paths"""
    paths = copy.deepcopy(paths)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    cache_file_dir = os.path.join(paths[0], f'cache_dims_{dims}_size_{img_size}_{eqv}.npz')
    gen_cache_file_dir = os.path.join(paths[1], f'cache_dims_{dims}_size_{img_size}_gen_Z2.npz')
    
    tmp_cache_img_dir = os.path.join(paths[1], "tmp")
    pathlib.Path(tmp_cache_img_dir).mkdir(parents=True, exist_ok=True)

    m1, s1 = calculate_statistics_of_path_cache(paths[0], cache_file_dir, model, batch_size, dims, img_size, device, num_workers, eqv)
    
    fid_values = []
    group_size = 1

    if eqv == 'Z2':
        group_size = 1
        def grp_op(x, k):
            return x
    elif eqv == 'V':
        group_size = 2
        def grp_op(x, k):
            return torch.cat([x, torch.flip(x, dims=[-2])], dim=0)
    elif eqv == 'H':
        group_size = 2
        def grp_op(x, k):
            return torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
    elif eqv == 'C4':    
        group_size = 4   
        def grp_op(x, k):
            return torch.cat([torch.rot90(x, k=k, dims=[-1, -2])], dim=0)
    elif eqv == 'D4':       
        group_size = 4
        def grp_op(x, k):
            return torch.cat(
                [torch.rot90(x, k=k, dims=[-1, -2])] + \
                [torch.rot90(torch.flip(x, dims=[-2]), k=k, dims = [-1, -2])], \
            dim=0)
    else:
        raise NotImplementedError(f"eqv: {eqv} is not implemented.")

    for i in range(0,group_size):
        path = pathlib.Path(paths[1])
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        dataset = ImagePathDataset(files, img_size, transforms=TF.ToTensor())
        to_image = TF.ToPILImage()
        print(dataset.__len__())
        for j in range(0,dataset.__len__()):
            image = grp_op(dataset.__getitem__(j), i)
            to_image(image).save(os.path.join(tmp_cache_img_dir, str(files[j])))
            
        m2, s2 = calculate_statistics_of_path(tmp_cache_img_dir, model, batch_size,
                                            dims, img_size, device, num_workers, eqv='Z2') # eqv='Z2' since we assume the model is generating a eqv invariant distribution

        fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
                          
    max_diff_fid_value = max(fid_values)-min(fid_values)

    return max_diff_fid_value


def main():
    print("\nCreating args parser...")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_fid_stats(paths=args.path,
                        batch_size=args.batch_size,
                        device=device,
                        dims=args.dims,
                        img_size=args.img_size,
                        num_workers=num_workers,
                        eqv=args.eqv)
        return

    print("Computing FID value...")
    fid_value = calculate_eqv_diff_fid_given_paths(paths=args.path,
                                          batch_size=args.batch_size,
                                          device=device,
                                          dims=args.dims,
                                          img_size=args.img_size,
                                          num_workers=num_workers,
                                          eqv=args.eqv)
    print('\nFID: ', fid_value)


if __name__ == '__main__':
    main()