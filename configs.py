config = {
    'nz': {'data_root': '/home/arc-zwq8528/Dataset/face_real_org2/train_set/',
            'data_lst': 'face_real_train_pair.txt',
            'mean_bgr': [104.00699, 116.66877, 122.67892],
            'yita': 0.5},
}

configs_test = {
    'face_real': {'data_root': 'E:\\Github\\Dataset/face_real/test_set/',
                'data_lst': 'face_real_test_pair.txt',
                'mean_bgr': [104.00699, 116.66877, 122.67892],
                'yita': 0.5},
}

if __name__ == '__main__':
    print(config.keys())
