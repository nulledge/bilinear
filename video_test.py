import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from model.hourglass import load
from util.preprocess_img import centerCrop
from util.euro_filter import OneEuroFilter


class CONFIG:
    nStacks = 8
    nFeatures = 256
    nModules = 1
    nJoints = 16
    nDepth = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parameter_dir = './save/SH_flip_center-moved/parameter'
    log_dir = './log/'


euro_cfg = {
    'freq': cap.get(cv.CAP_PROP_FPS),  # Hz
    'mincutoff': 1.0,  # FIXME
    'beta': 0.007,  # FIXME
    'dcutoff': 1.0  # this one should be ok
}
euro_filter = OneEuroFilter(**euro_cfg)
timestamp = 0

loaded_model, _, _, _ = load(CONFIG.device, CONFIG.parameter_dir, force_load=181)
loaded_model.train()

file_name = 'hummasong.mp4'
output_name = file_name.split('.')[0] + '_output' + '_StackedHourglass.mp4'
cap = cv.VideoCapture(file_name)
out = cv.VideoWriter(output_name, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv.CAP_PROP_FPS), (512, 256))

currentTime = 18000
cap.set(cv.CAP_PROP_POS_MSEC, currentTime)
isCaptured, frame_org = cap.read()

with tqdm(total=cap.get(cv.CAP_PROP_FRAME_COUNT), desc='valid epoch') as progress:
    while isCaptured:
        frame_org = centerCrop(frame_org)
        frame_org = cv.resize(frame_org, dsize=(256, 256))
        print(frame_org[128, 128, :])
        frame = np.swapaxes(np.swapaxes(frame_org, 1, 2), 0, 1)
        frame = np.divide(frame, 255)
        frameTensor = torch.Tensor([frame])
        pred = loaded_model(frameTensor.to(CONFIG.device))
        pred = torch.Tensor.cpu(pred[7]).detach().numpy()

        pred = euro_filter(pred, timestamp)
        timestamp += 1.0 / euro_cfg['freq']

        joint_pos = []
        for j in range(pred.shape[1]):
            tmp = pred[0, j, :, :]
            pos = np.unravel_index(tmp.argmax(), tmp.shape)
            pos = (pos[1] * 4, pos[0] * 4)
            joint_pos.append(pos)

        canvas_img = frame_org.copy()

        right_color = (0, 0, 255)  # right: red
        left_color = (255, 0, 0)  # left: blue
        spine_color = (0, 255, 0)  # spine: green

        cv.line(canvas_img, joint_pos[0], joint_pos[1], right_color, 5)
        cv.line(canvas_img, joint_pos[1], joint_pos[2], right_color, 5)
        cv.line(canvas_img, joint_pos[2], joint_pos[6], right_color, 5)

        cv.line(canvas_img, joint_pos[5], joint_pos[4], left_color, 5)
        cv.line(canvas_img, joint_pos[4], joint_pos[3], left_color, 5)
        cv.line(canvas_img, joint_pos[3], joint_pos[6], left_color, 5)

        cv.line(canvas_img, joint_pos[10], joint_pos[11], right_color, 5)
        cv.line(canvas_img, joint_pos[11], joint_pos[12], right_color, 5)
        cv.line(canvas_img, joint_pos[12], joint_pos[7], right_color, 5)

        cv.line(canvas_img, joint_pos[15], joint_pos[14], left_color, 5)
        cv.line(canvas_img, joint_pos[14], joint_pos[13], left_color, 5)
        cv.line(canvas_img, joint_pos[13], joint_pos[7], left_color, 5)

        cv.line(canvas_img, joint_pos[6], joint_pos[7], spine_color, 5)
        cv.line(canvas_img, joint_pos[7], joint_pos[8], spine_color, 5)
        cv.line(canvas_img, joint_pos[8], joint_pos[9], spine_color, 5)

        for j in range(pred.shape[1]):
            cv.circle(canvas_img, joint_pos[j], 3, (0, 255, 255), -1)  # yellow circle in joint

        concat_img = np.concatenate((frame_org, canvas_img), axis=1)
        out.write(concat_img)

        isCaptured, frame_org = cap.read()

        progress.update(1)
