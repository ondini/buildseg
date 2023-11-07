import fiftyone as fo
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-n', '--name', default='SEAAIDataset', type=str,
                        help='dataset name (default: SEAAIDataset)')
    parser.add_argument('-dp', '--port', default=5151, type=int,
                        help='port on which launch the dataset app (default: 5151)')

    args = parser.parse_args()

    dataset = fo.load_dataset(args.name)
    if args.port > 0:
        session = fo.launch_app(dataset, port=args.port, auto=False)
        session.wait()
