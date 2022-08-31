import argparse
import os
import time
from pathlib import Path
import cv2
import torch
from numpy import random
from ball_models.experimental import attempt_load
from ball_utils.datasets import LoadStreams, LoadImages
from ball_utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from ball_utils.torch_utils import select_device, time_synchronized
from deep_sort import build_tracker



def detect():
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    set_logging()
    device = select_device(opt.device)

    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    source_list = [source]
    for source in source_list:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
        deepsort = build_tracker('models/deep_sort.yaml', use_cuda=device.type != 'cpu')
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # inference
        # t0 for starting time
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        count = -1
        for path, img, im0s, vid_cap in dataset:
            count = count + 1
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s

                # video saving path and text saving path
                # print(save_path)
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)  # make new output folder
                # txt_path = save_path + source
                # save_path += source + '.mp4'
                save_path = out + "/" + Path(source).stem + '.mp4'

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                im_sort = im0.copy()
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # do tracking
                    bbox_xyxy, cls_conf = det[:, :4], det[:, 4]
                    bbox_xywh = xyxy2xywh(bbox_xyxy)
                    outputs = deepsort.update(bbox_xywh.cpu().numpy(), cls_conf.cpu().numpy(), im_sort)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        for output in outputs:
                            if save_img or view_img:  # Add bbox to image
                                label = str(output[-1])
                                plot_one_box(output[:4], im_sort, label=label, color=colors[0], line_thickness=3)

                    # Write results
                    txt_path = out + "/" + Path(source).stem + '.txt'
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (count, *xyxy))  # label format

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images' and False:
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            fourcc = 'mp4v'  # output video codec
                            fps = 25
                            h, w = im0.shape[:2]
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % Path(out))

        print('Process Done. (%.3fs)' % (time.time() - t0))
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()  # release previous video writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./model/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='', help='source')  # file/folder
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    print(f"weights = {opt.weights}")
    print(f"source = {opt.source}")
    print(f"output = {opt.output}")
    print(f"confidence threshold = {opt.conf_thres}")
    print(f"device = {opt.device}")
    print('-------------\n')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
