import logging
import torch
import os.path as osp
import torch.nn.functional as F
import time
import numpy as np

from optic.utils.file_io import save_list
from optic.engine import GradeTester
import optic.data.test_augment as ta

logger = logging.getLogger(__name__)


class MultimodalTester(GradeTester):
    @torch.no_grad()
    def test(self):
        self.reset_meter()
        self.model.eval()

        max_iter = len(self.test_loader)
        end = time.time()
        if self.cfg.test.save_prediction:
            save_predicts = []
        for i, sample in enumerate(self.test_loader):
            octs = self.preprocess_oct(sample["oct_img"])
            imgs = self.preprocess_image(sample["img"])
            #inputs = torch.from_numpy(inputs).to(self.device)
            label = sample["label"]
            # forward
            if isinstance(octs, list):
                outputs = [self.model(x) for x in zip(octs, imgs)]
                outputs = [torch.cat(x, dim=0) for x in outputs]
            else:
                outputs = self.model((octs, imgs))
            outputs = torch.cat(outputs, dim=0)
            predicts = F.softmax(outputs, dim=1)
            predicts = ta.fuse_predicts(predicts, reduce=self.cfg.test.augment.reduce)
            # avg_pred = predicts[0]
            pred_label = torch.argmax(predicts, dim=0)
            self.evaluator.update(
                np.expand_dims(pred_label.detach().cpu(), axis=0),
                np.expand_dims(label, axis=0)
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter)
            if self.cfg.test.save_prediction:
                predicts = predicts.tolist()
                save_predicts.append(
                    "{:04d},{:.5f},{:.5f},{:.5f}".format(
                        sample["id"], predicts[0], predicts[1], predicts[2]
                    )
                )
            end = time.time()
        self.log_epoch_info()
        if self.cfg.test.save_prediction:
            save_list(save_predicts, osp.join(self.work_dir, self.cfg.test.save_prediction))
