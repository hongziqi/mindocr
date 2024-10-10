import os
import sys
import logging

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

import mindspore as ms
import numpy as np
import cv2
import copy

from mindspore import Tensor
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.data.transforms import create_transforms, run_transforms
from typing import List
from structure_analyzer import StrutureAnalyzer
from matcher import TableMasterMatcher

sys.path.append('/home/candyhong/workspace/mindocr/tools/infer/text')
from tools.infer.text.predict_system import TextSystem
from tools.infer.text.config import parse_args

logger = logging.getLogger("docinsight.table")


class TableAnalyzer(object):
    def __init__(self, table_name: str, det_name=None, rec_name=None, **kwargs):
        """
        Initialize the table analyzer.
        Args:
            table_name (str): table model name
            det_name (str): text detection model name
            rec_name (str): text recognition model name
        """
        args = parse_args()
        if det_name:
            args.det_algorithm = det_name
        if rec_name:
            args.rec_algorithm = rec_name
        # print(f"args: {args}")
        self.text_system = TextSystem(args)
        self.table_structurer = StrutureAnalyzer(model_name=table_name, **kwargs)
        self.match = TableMasterMatcher()

    def _structure(self, img: np.ndarray):
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        return structure_res, elapse

    def _text_ocr(self, img: np.ndarray):
        logger.info(f"Start text detection and recognition.")
        boxes, text_scores, time_prof = self.text_system(copy.deepcopy(img))
        h, w = img.shape[:2]
        # Convert to format [x_min, y_min, x_max, y_max]
        r_boxes = []
        for box in boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        return dt_boxes, text_scores, time_prof

    def __call__(self, img: np.ndarray):
        boxes, text_scores, time_prof = self._text_ocr(img)
        structure_res, elapsed_time = self._structure(img)
        pred_html = self.match(structure_res, boxes, text_scores)
        return pred_html


def to_excel(html_table, excel_path):
    from tablepyxl import tablepyxl
    from bs4 import BeautifulSoup
    import openpyxl

    # 使用 BeautifulSoup 解析 HTML 字符串
    soup = BeautifulSoup(html_table, 'html.parser')

    table = soup.find('table')

    if not table:
        assert False, "No table found in the HTML string."

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)

    tablepyxl.document_to_xl(soup.prettify(), excel_path)


def to_csv(html_table, csv_path):
    from bs4 import BeautifulSoup
    import pandas as pd

    soup = BeautifulSoup(html_table, 'html.parser')

    table = soup.find('table')

    if not table:
        assert False, "No table found in the HTML string."

    df = pd.read_html(str(table))[0]
    df.to_csv(csv_path, index=False)


def main():
    analyzer = TableAnalyzer(table_name="table_master", det_name="DB_PPOCRv3", rec_name="CRNN")
    test_img_path = "docinsight/docs/table/table.jpg"
    img = cv2.imread(test_img_path)
    pred_html = analyzer(img)
    # to_excel(pred_html, "docinsight/output/table.xlsx")
    to_csv(pred_html, "docinsight/output/table.csv")


if __name__ == "__main__":
    main()
