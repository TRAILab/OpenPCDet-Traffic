import pickle

from ..kitti.kitti_dataset import KittiDataset


class TrafficDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )


def create_traffic_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = TrafficDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('traffic_infos_%s.pkl' % train_split)
    val_filename = save_path / ('traffic_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'traffic_infos_trainval.pkl'
    test_filename = save_path / 'traffic_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    traffic_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(traffic_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(traffic_infos_train + traffic_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    traffic_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(traffic_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_traffic_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_traffic_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Truck'],
            data_path=ROOT_DIR / 'data' / 'traffic-long',
            save_path=ROOT_DIR / 'data' / 'traffic-long'
        )
