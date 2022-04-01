import pandas as pd

from utils.core.inference import Inference
from configs.setting import global_setting
from model.model_load import load_ensemble
from dataload.dataloader import build_dataloader


def do_inference():
    # 모델 및 토크나이저
    config, device = global_setting('cfg.yaml')
    model_list, tokenizer = load_ensemble(config=config, device=device)

    # dataframes
    test_df = pd.read_csv('./data/test_data.csv')

    # 데이터 로더
    test_loader = build_dataloader(cfg=config, df=test_df, tokenizer=tokenizer, device=device, mode='test')

    inference = Inference(config=config,
                          model_list=model_list,
                          test_loader=test_loader,
                          device=device,
                          tokenizer=tokenizer)
    cls_label_map = test_loader.dataset.map_cls_to_label
    inference.inference(cls_label_map=cls_label_map)


if __name__ == '__main__':
    do_inference()

