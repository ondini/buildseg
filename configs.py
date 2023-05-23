model_weights = {
    'bl_bce_a084': '/home/kafkaon1/FVAPP/out/train/run_230517-134445/checkpoints/Deeplabv3_err:0.01578_ep:17.pth'
}

model_weights2 = {
    'wbce_a08': '/home/kafkaon1/FVAPP/out/train/run_230507-230929/checkpoints/Deeplabv3_err:0.04721_ep:15.pth',
    'bdc': '/home/kafkaon1/FVAPP/out/train/run_230508-224653/checkpoints/Deeplabv3_err:0.06075_ep:30.pth',
    'foc': '/home/kafkaon1/FVAPP/out/train/run_230510-093710/checkpoints/Deeplabv3_err:0.01155_ep:18.pth',
    'wbce_a05':'/home/kafkaon1/FVAPP/out/train/run_230510-233206/checkpoints/Deeplabv3_err:0.04527_ep:25.pth',
    'bl_bdc_a06': '/home/kafkaon1/FVAPP/out/train/run_230512-230005/checkpoints/Deeplabv3_err:0.12547_ep:30.pth',
    'bl_bdc_a08_enc': '/home/kafkaon1/FVAPP/out/train/run_230514-141833/checkpoints/Deeplabv3_err:0.14559_ep:30.pth',
    'bl_bdc_a084': '/home/kafkaon1/FVAPP/out/train/run_230514-141833/checkpoints/Deeplabv3_err:0.14559_ep:30.pth',
    'bce': '/home/kafkaon1/FVAPP/out/train/run_230516-175052/checkpoints/Deeplabv3_err:0.05070_ep:21.pth',
    'bce_enc': '/home/kafkaon1/FVAPP/out/train/run_230516-183515/checkpoints/Deeplabv3_err:0.04982_ep:25.pth',
    'wbce_a02_enc': '/home/kafkaon1/FVAPP/out/train/run_230517-134445/checkpoints/Deeplabv3_err:0.01578_ep:17.pth',
}

if __name__ == '__main__':
    print(model_weights.keys())
