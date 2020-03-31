import cv2
import numpy as np
import math

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def py_nms(boxes, scores, max_boxes=80, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]

def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=np.float32)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def post_process(detections, org_img_shape, input_size, down_ratio, score_threshold):
    bboxes = detections[0, :, 0:4]
    scores = detections[0, :, 4]
    classes = detections[0, :, 5]
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size[1] / org_w, input_size[0] / org_h)

    dw = (input_size[1] - resize_ratio * org_w) / 2
    dh = (input_size[0] - resize_ratio * org_h) / 2

    bboxes[:, 0::2] = 1.0 * (bboxes[:, 0::2] * down_ratio - dw) / resize_ratio
    bboxes[:, 1::2] = 1.0 * (bboxes[:, 1::2] * down_ratio - dh) / resize_ratio
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, org_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, org_h)
    score_mask = scores >= score_threshold
    bboxes, socres, classes = bboxes[score_mask], scores[score_mask], classes[score_mask]
    return np.concatenate([bboxes, socres[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def bboxes_draw_on_img(img, classes_id, scores, bboxes, class_names, thickness=2):
    colors_tableau = [(158, 218, 229), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207)]
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors_tableau[int(classes_id[i])]
        # Draw bounding boxes
        x1_src = int(bbox[0])
        y1_src = int(bbox[1])
        x2_src = int(bbox[2])
        y2_src = int(bbox[3])

        cv2.rectangle(img, (x1_src, y1_src), (x2_src, y2_src), color, thickness)
        # Draw text
        s = '%s: %.2f' % (class_names[int(classes_id[i])], scores[i])
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (y1_src - text_size[1], x1_src)

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)

def get_preds_gpu(detections, image_id):
    '''
    Given the y_pred of an input image, get the predicted bbox and label info.
    return:
        pred_content: 2d list.
    '''

    cls_in_img = list(set(detections[:, 5]))
    results = []
    pred_content = []
    for c in cls_in_img:
        cls_mask = (detections[:, 5] == c)
        classified_det = detections[cls_mask]
        classified_bboxes = classified_det[:, :4]
        classified_scores = classified_det[:, 4]
        inds = py_nms(classified_bboxes, classified_scores, max_boxes=50, iou_thresh=0.5)
        # results.extend(classified_det[inds])
        results.extend(classified_det[inds].tolist())
    for bbox in results:
        import cfg
        if bbox[4] >= cfg.score_threshold:
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            score = float(bbox[4])
            label = int(bbox[5])
            pred_content.append([image_id, x_min, y_min, x_max, y_max, score, label])
    # print("results is", results)
    # print("pred_content is", pred_content)
    return pred_content

def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.
    line format: line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
    return:
        line_idx: int64
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
        labels: shape [N]. class index.
        img_width: int.
        img_height: int
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    line_idx = s[0]
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    box_cnt = len(s)
    assert box_cnt > 0
    boxes = []
    labels = []
    gt_labels = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in s])
    # print(labels)
    for idx, label in enumerate(gt_labels):
        # box = label[:4]
        # class_name = label[4]
        class_name, x_min, y_min, x_max, y_max = label[4], label[0], label[1], label[2], label[3]
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(class_name)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.float32)
    return  pic_path, boxes, labels, img_width, img_height

gt_dict = {}  # key: img_id, value: gt object list
def parse_gt_rec(gt_filename, target_img_size, letterbox_resize=True):
    '''
    parse and re-organize the gt info.
    return:
        gt_dict: dict. Each key is a img_id, the value is the gt bboxes in the corresponding img.
    '''

    global gt_dict

    if not gt_dict:
        new_width, new_height = target_img_size
        with open(gt_filename, 'r') as f:
            for img_id, line in enumerate(f):
                pic_path, boxes, labels, ori_width, ori_height = parse_line(line)

                objects = []
                for i in range(len(labels)):
                    x_min, y_min, x_max, y_max = boxes[i]
                    label = labels[i]

                    if letterbox_resize:
                        resize_ratio = min(new_width / ori_width, new_height / ori_height)

                        resize_w = int(resize_ratio * ori_width)
                        resize_h = int(resize_ratio * ori_height)

                        dw = int((new_width - resize_w) / 2)
                        dh = int((new_height - resize_h) / 2)

                        objects.append([x_min * resize_ratio + dw,
                                        y_min * resize_ratio + dh,
                                        x_max * resize_ratio + dw,
                                        y_max * resize_ratio + dh,
                                        label])
                    else:
                        # objects.append([x_min * new_width / ori_width,
                        #                 y_min * new_height / ori_height,
                        #                 x_max * new_width / ori_width,
                        #                 y_max * new_height / ori_height,
                        #                 label])
                        objects.append([x_min, y_min, x_max, y_max , label])
                gt_dict[img_id] = objects
    return gt_dict


# The following two functions are modified from FAIR's Detectron repo to calculate mAP:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(gt_dict, val_preds, classidx, iou_thres=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''
    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        bbox = np.array([x[:4] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # 2. obtain pred results
    pred = [x for x in val_preds if x[-1] == classidx]
    img_ids = [x[0] for x in pred]
    confidence = np.array([x[-2] for x in pred])
    BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])

    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    try:
        BB = BB[sorted_ind, :]
    except:
        print('no box, ignore')
        return 1e-6, 1e-6, 0, 0, 0
    img_ids = [img_ids[x] for x in sorted_ind]

    # 4. mark TPs and FPs
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # all the gt info in some image
        R = class_recs[img_ids[d]]
        bb = BB[d, :]
        ovmax = -np.Inf
        BBGT = R['bbox']

        if BBGT.size > 0:
            # calc iou
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
                        BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thres:
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    return npos, nd, tp[-1] / float(npos), tp[-1] / float(nd), ap
