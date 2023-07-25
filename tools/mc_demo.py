import sys
import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import json
import glob
import pandas as pd

import torchvision.transforms
from loguru import logger

sys.path.append('.')

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

from HRNet.model import HighResolutionNet
from HRNet.draw_utils import draw_keypoints
from HRNet import transforms

from PPE.pose_utils import process_bbox, pixel2cam
from PPE.dataset import generate_patch_image
from PPE.config import cfg
# from torchvision import transforms
from torch.nn.parallel.data_parallel import DataParallel
from PPE.model import get_pose_net
from PPE.vis import vis_3d_multiple_skeleton
from PPE.plots import Annotator

from SVHNmodel.SVHN_model import Multi_Digit, process_image

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use reid model")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre) # 把预测类别的one-hot向量转化为int类型
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    tracker = BoTSORT(args, frame_rate=args.fps)

    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):

        # Detect objects
        outputs, img_info = predictor.inference(img_path, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        detections = []
        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

        # Run tracker
        online_targets = tracker.update(detections, img_info['raw_img'])

        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > args.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_cls.append(t.cls)

                # save results
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
        timer.toc()
        online_im = plot_tracking(
            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time, ids2=online_cls
        )
        # else:
        #     timer.toc()
        #     online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def loadMutidigitModel():
    MultidigitModel = Multi_Digit() #input: 1*100*100
    modelweight = torch.load('SVHNmodel/multidigit.pth')
    modelweight = modelweight['net']
    MultidigitModel.load_state_dict(modelweight)
    MultidigitModel.cuda()
    MultidigitModel.eval()
    return MultidigitModel


def loadHRNet():

    keypoint_json_path = "HRNet/person_keypoints.json"
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)
    skeleton = person_info['skeleton']
    HPEmodel = HighResolutionNet(base_channel=32)
    HPEweights = torch.load("HRNet/pytorch/pose_coco/pose_hrnet_w32_256x192.pth")
    HPEweights = HPEweights if "model" not in HPEweights else HPEweights["model"]
    HPEmodel.load_state_dict(HPEweights)
    HPEmodel.cuda()
    HPEmodel.eval()
    resize_hw = (256, 192)
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return HPEmodel, data_transform, skeleton


def load3DMPPE():
    # 3DMPPE
    joint_num = 21
    focal = [1500, 1500]
    joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
    flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
    skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20),
        (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
    skeleton = np.array([[i1 + 1, i2 + 1] for i1, i2 in skeleton])

    model_path = 'PPE/snapshot_%d.pth.tar' % int(24)
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    PPEmodel = get_pose_net(cfg, False, joint_num)
    PPEmodel = DataParallel(PPEmodel).cuda()
    PPEckpt = torch.load(model_path)
    PPEmodel.load_state_dict(PPEckpt['network'])
    PPEmodel.eval()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    return PPEmodel, transform, focal, skeleton


def iou_tlwh(a: np.ndarray, b: np.ndarray) -> float:
    a_tl = a[:2]
    a_br = np.array([a[0]+a[2], a[1]+a[3]])
    b_tl = b[:2]
    b_br = np.array([b[0]+b[2], b[1]+b[3]])
    int_tl = np.maximum(a_tl, b_tl)
    int_br = np.minimum(a_br, b_br)
    a_area = np.product(a_br - a_tl)
    b_area = np.product(b_br - b_tl)
    int_area = np.product(np.maximum(0., int_br - int_tl))
    return int_area / (a_area + b_area - int_area)


def imageflow_demo(predictor, vis_folder, current_time, args, helmet_data):
    filename = args.path.split('/')[-1]
    filename = filename.split('.')[:-1]
    game = int(filename[0].split('_')[0])
    play = int(filename[0].split('_')[1])
    view = filename[0].split('_')[2]
    id_play = (helmet_data['game_key'] == game) * (helmet_data['play_id'] == play) * (helmet_data['view'] == view)
    frame_id_start = helmet_data.loc[id_play, 'frame'].min()
    frame_id_end = helmet_data.loc[id_play, 'frame'].max()

    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    vid_writer_skeleton = cv2.VideoWriter(
        f"{save_path.split('.mp4')[0]}_skeleton.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BoTSORT(args, frame_rate=args.fps)
    timer = Timer()
    frame_id = 1
    results = []

    """load 3DMPPE"""
    # PPEmodel, transform, focal, skeleton = load3DMPPE()

    """load HRNet"""
    HPEmodel, data_transform, skeleton = loadHRNet()

    # """load Multi digit model"""
    # MultidigitModel = loadMutidigitModel()

    results_pd = pd.DataFrame(columns=['game', 'play', 'view', 'player_id', 'frame_id', 'bbox_left', 'bbox_top',
                                       'bbox_right', 'bbox_bottom',
                                       "nose_x", "nose_y",
                                       "left_eye_x", "left_eye_y",
                                       "right_eye_x", "right_eye_y",
                                       "left_ear_x", "left_ear_y",
                                       "right_ear_x", "right_ear_y",
                                       "left_shoulder_x", "left_shoulder_y",
                                       "right_shoulder_x", "right_shoulder_y",
                                       "left_elbow_x", "left_elbow_y",
                                       "right_elbow_x", "right_elbow_y",
                                       "left_wrist_x", "left_wrist_y",
                                       "right_wrist_x", "right_wrist_y",
                                       "left_hip_x", "left_hip_y",
                                       "right_hip_x", "right_hip_y",
                                       "left_knee_x", "left_knee_y",
                                       "right_knee_x", "right_knee_y",
                                       "left_ankle_x", "left_ankle_y",
                                       "right_ankle_x", "right_ankle_y"])
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if frame_id < frame_id_start:
            frame_id += 1
            continue
        if frame_id > frame_id_end:
            break
        if ret_val:
            # Detect objects
            outputs, img_info = predictor.inference(frame, timer)
            scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

            detections = []
            if outputs[0] is not None:
                outputs = outputs[0].cpu().numpy()
                detections = outputs[:, :7]
                detections[:, :4] /= scale

            # Run tracker
            online_targets = tracker.update(detections, img_info["raw_img"])

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()

            """姿态估计"""
            annotator_raw = Annotator(img_info['raw_img'].copy(), line_width=3)
            imb = img_info['raw_img'].copy()
            imb.fill(0)
            annotator_blank = Annotator(imb, line_width=3)

            id_frame = id_play * (helmet_data['frame'] == frame_id)
            helmet_data_frame = helmet_data.loc[id_frame]
            helmet_tlwhs = helmet_data_frame[['left', 'top', 'width', 'height']].to_numpy()
            helmet_ids = helmet_data_frame['player_label'].to_numpy()
            head_tlwh_list = []
            player_id_list = []
            player_label_list = []
            i_dels = []
            for i, (tlwh, conf) in enumerate(zip(online_tlwhs, online_scores)):
                tlwh[2:] = tlwh[2:] * 1.02 + 10
                x1, y1, w, h = tlwh
                x1 = max(0, x1)
                y1 = max(0, y1)
                xyxy1 = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
                xyxy1 = np.array(xyxy1)
                crop = img_info['raw_img'][xyxy1[1]:xyxy1[3], xyxy1[0]:xyxy1[2], ::-1]

                # """Multidigit detection"""
                # print("")
                # crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                # crop_gray = process_image(crop_gray)
                # tran = torchvision.transforms.ToTensor()
                # tran2 = torchvision.transforms.Normalize(mean=0.5, std=0.5)
                # crop_gray = tran2(tran(crop_gray))
                # digitoutput = MultidigitModel(crop_gray.cuda())
                # number1 = np.argmax(digitoutput[0, 0:11].detach().cpu()) - 1
                # number2 = np.argmax(digitoutput[0, 11:22].detach().cpu()) - 1
                # cv2.imwrite('test.jpg', crop_gray.numpy().squeeze() * 255)
                # """Multidigit detection"""

                """HRNet"""
                # xyxy1 = torch.tensor(xyxy).view(-1, 4)
                # b = xyxy2xywh(xyxy1)  # boxes
                # b[:, 2:] = b[:, 2:] * 1.02 + 10  # box wh * gain + pad
                # xyxy1 = xywh2xyxy(b).long()
                img_tensor, target = data_transform(crop, {"box": [0, 0, crop.shape[1] - 1, crop.shape[0] - 1]})
                img_tensor = torch.unsqueeze(img_tensor, dim=0)
                outputs = HPEmodel(img_tensor.cuda())

                flip_test = False
                if flip_test:
                    flip_tensor = transforms.flip_images(img_tensor)
                    flip_outputs = torch.squeeze(
                        transforms.flip_back(HPEmodel(flip_tensor.cuda()), person_info["flip_pairs"]),
                    )
                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                    flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                    outputs = (outputs + flip_outputs) * 0.5

                keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
                keypoints = np.squeeze(keypoints)
                scores = np.squeeze(scores)

                head_tlwh = [int(x1 + keypoints[0, 0]) - 15, int(y1 + keypoints[0, 1]) - 15, 30, 30]
                head_tlwh_list.append(head_tlwh)
                helmet_head_iou = [iou_tlwh(head_tlwh, helmet_tlwh) for helmet_tlwh in helmet_tlwhs]

                if np.max(helmet_head_iou) == 0:
                    i_dels.append(i)
                    continue

                helmet_id = np.argmax(helmet_head_iou)
                player_id = helmet_data_frame.iloc[helmet_id]['nfl_player_id']
                player_label = helmet_data_frame.iloc[helmet_id]['player_label']
                player_id_list.append(player_id)
                player_label_list.append(player_label)
                """HRNet"""

                """3DMPPE姿态估计"""
                # output_pose_3d_list = []
                # # mask_c = masks[int(xyxy1[0, 1]):int(xyxy1[0, 3]), int(xyxy1[0, 0]):int(xyxy1[0, 2]), j] #segment
                # # crop_mask = crop * mask_c[...,None] #segment
                # # crop = crop_mask #segment
                # original_img_height, original_img_width = crop.shape[:2]
                # princpt = [original_img_width / 2, original_img_height / 2]  # x-axis, y-axis
                # bbox_list = [0.0, 0.0, original_img_width, original_img_height]
                # root_depth_list = [20000]
                # bbox = process_bbox(np.array(bbox_list), original_img_width, original_img_height)
                # img, img2bb_trans = generate_patch_image(crop, bbox, False, 1.0, 0.0, False)
                # img = transform(img).cuda()[None, :, :, :]
                #
                # # forward
                # with torch.no_grad():
                #     pose_3d = PPEmodel(img)  # x,y: pixel, z: root-relative depth (mm)
                # # inverse affine transform (restore the crop and resize)
                # pose_3d = pose_3d[0].cpu().numpy()
                # pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
                # pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
                # pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
                # img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
                # pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001),
                #                         pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                # # output_pose_2d_list.append(pose_3d[:, :2].copy())
                # keypoints = pose_3d.copy()
                #
                # # root-relative discretized depth -> absolute continuous depth
                # pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + \
                #                 root_depth_list[0]
                # pose_3d = pixel2cam(pose_3d, focal, princpt)
                # output_pose_3d_list.append(pose_3d.copy())
                # # visualize 3d poses
                # vis_kps = np.array(output_pose_3d_list)
                # # vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton,
                # #                              'output_pose_3d (x,y,z: camera-centered. mm.)')
                """3DMPPE姿态估计"""

                annotator_raw.draw_skeleton(xyxy1, keypoints, skeleton, scores=None, thresh=0.2, r=3)
                annotator_blank.draw_skeleton(xyxy1, keypoints, skeleton, scores=None, thresh=0.2, r=3)
                im_raw = annotator_raw.result()
                im_blank = annotator_blank.result()
                keypoints_raw = keypoints.copy()
                keypoints_raw[:, 0] = keypoints[:, 0] + xyxy1[0]
                keypoints_raw[:, 1] = keypoints[:, 1] + xyxy1[1]
                keypoints_raw = keypoints_raw.reshape([1, -1]).squeeze().astype('int')
                result_list = [game, play, view, player_id, frame_id]
                result_list.extend(xyxy1)
                result_list.extend(keypoints_raw)
                results_pd.loc[len(results_pd)] = result_list
            """姿态估计"""

            # online_im = plot_tracking(
            #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            # )
            head_tlwh_list = np.array(head_tlwh_list)

            i_dels.reverse()
            for i_del in i_dels:
                online_tlwhs.pop(i_del)
            online_im_raw = plot_tracking(
                im_raw, online_tlwhs, player_label_list, frame_id=frame_id, fps=1. / timer.average_time, drawbox=True,
                mycolor=[0, 0, 255])
            online_im_blank = plot_tracking(
                im_blank, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time, drawbox=False
            )



            # online_im_raw = plot_tracking(
            #     online_im_raw, helmet_tlwhs, helmet_ids, frame_id=frame_id, fps=1. / timer.average_time,
            #     put_text=False, mycolor=[0, 255, 0])
            # else:
            #     timer.toc()
            #     online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im_raw)
                vid_writer_skeleton.write(online_im_blank)
            # cv2.imshow('1',online_im_raw)
            # cv2.imshow('2',online_im_blank)
            ch = cv2.waitKey(1)
            if ch == 32:
                cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
        res_file_csv = osp.join(vis_folder, f"{timestamp}", f"{filename[0]}.csv")
        results_pd.to_csv(res_file_csv)

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()

    files = glob.glob(args.path)
    num_files = len(files)
    helmet_data = pd.read_csv('data/train_baseline_helmets.csv')
    for i, file in enumerate(files[32:]):
        logger.info(f"processing video {i+1} / {num_files}")
        args.path = file
        if args.demo == "image" or args.demo == "images":
            image_demo(predictor, vis_folder, current_time, args)
        elif args.demo == "video" or args.demo == "webcam":
            imageflow_demo(predictor, vis_folder, current_time, args, helmet_data)
        else:
            raise ValueError("Error: Unknown source: " + args.demo)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args.ablation = False
    args.mot20 = not args.fuse_score

    main(exp, args)
