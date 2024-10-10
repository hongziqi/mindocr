import os
import sys
import time
import logging

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

import mindspore as ms
import numpy as np
import cv2

from mindspore import Tensor
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.data.transforms import create_transforms, run_transforms
from typing import List
from docinsight.utility import download_ckpt

logger = logging.getLogger("docinsight.table")

support_models = {
    "table_master":
        {
            "ckpt_url": "https://download-mindspore.osinfra.cn/toolkits/mindocr/tablemaster/table_master-78bf35bb.ckpt",
            "config": {
                "type": "table",
                "transform": None,
                "backbone": {
                    "name": "table_resnet_extra",
                    "gcb_config": {
                        "ratio": 0.0625,
                        "headers": 1,
                        "att_scale": False,
                        "fusion_type": "channel_add",
                        "layers": [False, True, True, True],
                    },
                    "layers": [1, 2, 5, 3],
                },
                "head": {
                    "name": "TableMasterHead",
                    "out_channels": 43,
                    "hidden_size": 512,
                    "headers": 8,
                    "dropout": 0.0,
                    "d_ff": 2024,
                    "max_text_length": 500,
                    "loc_reg_num": 4
                },
            }
        }
}


class StrutureAnalyzer(object):
    """
    Infer model for table structure analysis

    Args:
        model (Model): model of table structure analysis
        preprocess_ops (Preprocessor): preprocess operations
        postprocess_ops (Postprocessor): postprocess operations

    Example:
        >>> analyzer = StrutureAnalyzer(model_name="table_master")
        >>> test_img = "docinsight/docs/table/table.jpg"
        >>> result = analyzer(test_img)

    """

    def __init__(self, model_name: str, model_config=None, ckpt_load_path=None, preprocess_list=None,
                 postprocess_dict=None, **kwargs):
        """
        Initialize the StrutureAnalyzer

        Args:
            model_name (str): model name
            model_config (dict): model config
            ckpt_load_path (str): checkpoint load path
            preprocess_list (List[dict]): preprocess operations
            postprocess_dict (dict): postprocess operations
            **kwargs: other arguments
        """
        if model_name in support_models.keys() and not model_config:
            model_config = support_models[model_name]["config"]
        assert model_config, f"model_config is required for model {model_name}"

        if model_name in support_models.keys() and not ckpt_load_path:
            ckpt_load_path = download_ckpt(support_models[model_name]["ckpt_url"])
        assert os.path.exists(ckpt_load_path) == True, f"ckpt_load_path {ckpt_load_path} not exists"

        max_len = kwargs.get("max_len", 480)
        amp_level = kwargs.get("amp_level", "O2")
        if ms.get_context("device_target") == "GPU" and amp_level == "O3":
            logger.warning(
                "Detection model prediction does not support amp_level O3 on GPU currently. "
                "The program has switched to amp_level O2 automatically."
            )
            amp_level = "O2"

        if not preprocess_list:
            preprocess_list = [
                {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                {"ResizeTableImage": {"max_len": max_len}},
                {"PaddingTableImage": {"size": [max_len, max_len]}},
                {
                    "TableImageNorm": {
                        "std": [0.5, 0.5, 0.5],
                        "mean": [0.5, 0.5, 0.5],
                        "scale": "1./255.",
                        "order": "hwc",
                    }
                },
                {"ToCHWImage": None}
            ]

        if not postprocess_dict:
            postprocess_dict = {
                "name": "TableMasterLabelDecode",
                "character_dict_path": "mindocr/utils/dict/table_master_structure_dict.txt",
                "merge_no_span_structure": True,
                "box_shape": "pad",
            }

        # init model
        self.model = build_model(model_config, ckpt_load_path=ckpt_load_path, amp_level=amp_level)
        self.model.set_train(False)
        self.preoprocess_ops = create_transforms(preprocess_list)
        self.postprocess_ops = build_postprocess(postprocess_dict)

    def preprocess(self, img_or_path):
        """
        Preprocess the input image
        """
        if isinstance(img_or_path, str):
            data = {"img_path": img_or_path}
            output = run_transforms(data, self.preoprocess_ops)
        elif isinstance(img_or_path, dict):
            output = run_transforms(img_or_path, self.preoprocess_ops)
        else:
            data = {"image": img_or_path}
            data["image_ori"] = img_or_path.copy()
            data["image_shape"] = img_or_path.shape
            output = run_transforms(data, self.preoprocess_ops[1:])
        return output

    def __call__(self, img_or_path):
        """
        Infer the model
        """
        starttime = time.time()
        data = self.preprocess(img_or_path)
        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            input_np = Tensor(np.expand_dims(input_np, axis=0))
        net_pred = self.model(input_np)
        preds = {}
        preds["structure_probs"] = net_pred[0]
        preds["loc_preds"] = net_pred[1]
        shape_list = np.expand_dims(data["shape"], axis=0)
        post_result = self.postprocess_ops(net_pred, [shape_list])

        structure_str_list = post_result['structure_batch_list'][0][0]
        structure_str_list = (
                ["<html>", "<body>", "<table>"]
                + structure_str_list
                + ["</table>", "</body>", "</html>"]
        )
        bbox_list = post_result['bbox_batch_list'][0]
        elapse = time.time() - starttime
        return (structure_str_list, bbox_list), elapse


def draw_rectangle(img_path, boxes):
    boxes = np.array(boxes)
    img = cv2.imread(img_path)
    img_show = img.copy()
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return img_show


def main():
    analyzer = StrutureAnalyzer(model_name="table_master")
    test_img_path = "docinsight/docs/table/table.jpg"
    test_img = cv2.imread(test_img_path)
    (structure_str_list, bbox_list), elapse = analyzer(test_img)
    # or use with a img path
    # (structure_str_list, bbox_list), elapse = analyzer(test_img_path)
    img = draw_rectangle(test_img_path, bbox_list)
    cv2.imwrite("docinsight/output/table.jpg", img)


if __name__ == "__main__":
    main()
