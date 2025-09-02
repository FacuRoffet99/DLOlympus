import numpy as np

from mmdet.evaluation import bbox_overlaps
from mmengine.utils import ProgressBar


def collect_detections_and_ground_truths(dataset,
										 results,
										 score_thr=0.0,
										 tp_iou_thr=0.5,
										 max_dets=300):
	"""
	Collect predicted and ground truth classes for each image.

	Args:
		dataset (Dataset): Test or val dataset.
		results (list[dict]): A list of detection results per image.
		score_thr (float): Score threshold to filter bboxes.
		tp_iou_thr (float): IoU threshold to be considered a TP.
		max_dets (int): Max detections per image (usually 100 for COCO mAP).

	Returns:
		tuple: (list of predicted classes, list of ground truth classes)
	"""
	all_pred_classes = []
	all_gt_classes = []

	assert len(dataset) == len(results)
	prog_bar = ProgressBar(len(results))

	for idx, per_img_res in enumerate(results):
		result = per_img_res['pred_instances']
		gts = dataset.get_data_info(idx)['instances']

		gt_bboxes = np.array([gt['bbox'] for gt in gts], dtype=np.float32)
		gt_labels = np.array([gt['bbox_label'] for gt in gts], dtype=np.int64)

		det_bboxes = result['bboxes'].numpy()
		det_scores = result['scores'].numpy()
		det_labels = result['labels'].numpy()

		# Filter by score threshold
		keep = det_scores >= score_thr
		det_bboxes = det_bboxes[keep]
		det_scores = det_scores[keep]
		det_labels = det_labels[keep]

		# Keep top-k detections (COCO default: 100)
		if len(det_scores) > max_dets:
			topk_inds = np.argsort(det_scores)[::-1][:max_dets]
			det_bboxes = det_bboxes[topk_inds]
			det_scores = det_scores[topk_inds]
			det_labels = det_labels[topk_inds]

		# No predictions or GTs
		if len(gt_bboxes) == 0 and len(det_bboxes) == 0:
			prog_bar.update()
			continue
		elif len(gt_bboxes) == 0:
			# All predictions are false positives
			all_pred_classes.extend(det_labels.tolist())
			all_gt_classes.extend([9999] * len(det_labels))
			prog_bar.update()
			continue
		elif len(det_bboxes) == 0:
			# All GTs are false negatives
			all_pred_classes.extend([9999] * len(gt_labels))
			all_gt_classes.extend(gt_labels.tolist())
			prog_bar.update()
			continue

		# IoU matrix
		ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
		matched_gts = set()
		matched_preds = set()

		for i in range(len(det_bboxes)):
			best_iou = 0
			best_gt_idx = -1
			for j in range(len(gt_bboxes)):
				if j in matched_gts:
					continue
				if ious[i, j] >= tp_iou_thr and ious[i, j] > best_iou:
					best_iou = ious[i, j]
					best_gt_idx = j
			if best_gt_idx >= 0:
				matched_gts.add(best_gt_idx)
				matched_preds.add(i)
				all_pred_classes.append(det_labels[i])
				all_gt_classes.append(gt_labels[best_gt_idx])

		# False positives (preds not matched)
		for i in range(len(det_bboxes)):
			if i not in matched_preds:
				all_pred_classes.append(det_labels[i])
				all_gt_classes.append(9999)

		# False negatives (GTs not matched)
		for j in range(len(gt_bboxes)):
			if j not in matched_gts:
				all_pred_classes.append(9999)
				all_gt_classes.append(gt_labels[j])

		prog_bar.update()

	return all_pred_classes, all_gt_classes