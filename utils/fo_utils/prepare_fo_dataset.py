import fiftyone as fo
import fiftyone.core.labels as fol
import random
import argparse

random.seed(51)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-d', '--dataset_path', default=None, type=str,
                        help='dataset path (default: None)')
    parser.add_argument('-n', '--name', default='SEAAIDataset', type=str,
                        help='dataset name (default: SEAAIDataset)')
    parser.add_argument('-s', '--split', default=0.2, type=float,
                        help='split between train and test (0=only train) (default: 0.2)')

    args = parser.parse_args()

    if not args.dataset_path is  None:
        dataset = fo.Dataset.from_dir(
            dataset_dir=args.dataset_path,
            dataset_type=fo.types.FiftyOneDataset,
            persistent = True,
            name = args.name
        )
        dataset.save()
    else:
        dataset = fo.load_dataset(args.name)

    for sample in dataset:
        if random.random() > args.split:
            sample.tags.append("train")  # Tag as "train"
        else:
            sample.tags.append("test")  # Tag as "test"
        sample.save()

    train_view = dataset.match_tags("train")
    dataset.save_view("train-split", train_view)

    train_view = dataset.match_tags("test")
    dataset.save_view("test-split", train_view)
