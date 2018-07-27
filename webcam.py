import cv2 as cv

import numpy as np
import torch

import Model.hourglass
from Model.end2end import softargmax
from util import config

device = torch.device(config.device)
hourglass, _, _, _, _ = Model.hourglass.load_model(device, config.pretrained['hourglass'])

cap = cv.VideoCapture(0)

with torch.set_grad_enabled(False):
    while (True):
        ret, frame = cap.read()

        image = cv.resize(frame, (256, 256))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.asarray(image / 255).astype(np.float32)  # HWC

        image = image.transpose(2, 0, 1)  # CHW
        image = torch.Tensor([image])  # BCHW
        image = image.to(device)

        output = hourglass(image)  #SBCHW

        image = np.asarray(image.data[0])  # CHW
        heatmaps = np.asarray(output.data[-1][0])  # CHW

        for joint, heatmap in enumerate(heatmaps):
            x, y = softargmax(torch.Tensor(heatmap).cuda())  # [0, 64)
            x, y = (int(x), int(y))
            for tx in range(-5, 5):
                for ty in range(-5, 5):
                    xx = 4*x + tx  # [0, 256)
                    yy = 4*y + ty  # [0, 256)
                    image[:, yy, xx] = [1, 0, 0]

        image = np.asarray(image * 255).astype(np.uint8).transpose(1, 2, 0)  #HWC
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = cv.resize(image.copy(), (1024, 1024))
        cv.imshow('frame', image)

        if cv.waitKey(1) and 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
