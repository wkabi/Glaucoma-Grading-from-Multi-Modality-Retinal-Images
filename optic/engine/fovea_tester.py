import logging
import os.path as osp
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

from optic.utils.misc import round_dict
from optic.utils.file_io import save_list
from optic.engine import GradeTester
from optic.evaluation import FoveaEvaluator, AverageMeter

logger = logging.getLogger(__name__)


class FoveaTester(GradeTester):
    def init_meter(self):
        self.evaluator = FoveaEvaluator()
        self.batch_time_meter = AverageMeter()
        logger.info("Meters initialized")

    def log_epoch_info(self):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        logger.info(
            "Test Epoch\t{}".format(json.dumps(round_dict(log_dict)))
        )

    @torch.no_grad()
    def test(self):
        self.reset_meter()
        self.model.eval()

        max_iter = len(self.test_loader)
        end = time.time()
        if self.cfg.test.save_prediction:
            save_predicts = []
        for i, samples in enumerate(self.test_loader):
            orig_height, orig_width = samples["img"].shape[:2]
            inputs = self.preprocess_image(samples["img"])
            labels = np.array([
                samples["fovea_x"] / orig_width,
                samples["fovea_y"] / orig_height,
                samples["has_fovea"]
            ])
            labels = np.expand_dims(labels, axis=0)
            labels = torch.from_numpy(labels).float().to(self.device)
            # forward
            # import ipdb; ipdb.set_trace()
            if isinstance(inputs, list):
                outputs = [self.model(x) for x in inputs]
                valid = []
                for o in outputs:
                    if o[0][2] > 0:
                        valid.append(o)
                outputs = torch.cat(valid, dim=0)
                outputs[:, 2] = F.sigmoid(outputs[:, 2])
                # weight = outputs[:, 2] / outputs[:, 2].sum()
                # pred_coord = torch.einsum("ij,i->ij", outputs[:, :2], weight).sum(dim=0)
                pred_coord = outputs[:, :2].mean(dim=0)
                pred_has_fovea_prob = outputs[:, 2].mean(dim=0)
                pred_has_fovea = (pred_has_fovea_prob > 0.5).float().view(1)
                preds = torch.cat((pred_coord.unsqueeze(0), pred_has_fovea.unsqueeze(0)), dim=1)
            else:
                outputs = self.model(inputs)
                pred_coord = outputs[:, :2]
                pred_has_fovea_prob = F.sigmoid(outputs[:, 2])
                pred_has_fovea = (pred_has_fovea_prob > 0.5).float()
                preds = torch.cat((pred_coord, pred_has_fovea.unsqueeze(1)), dim=1)
            self.evaluator.update(
                preds.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter)
            if self.cfg.test.save_prediction:
                pred_coord = pred_coord.tolist()
                pred_has_fovea_prob = pred_has_fovea_prob.tolist()
                save_predicts.append(
                    "{:04d},{:.5f},{:.5f},{:.5f},{},{}".format(
                        samples["id"], pred_coord[0], pred_coord[1],
                        pred_has_fovea_prob, orig_width, orig_height
                    )
                )
        self.log_epoch_info()
        if self.cfg.test.save_prediction:
            save_list(save_predicts, osp.join(self.work_dir, self.cfg.test.save_prediction))
