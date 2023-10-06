import os
import cv2
import time
import numpy as np
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.age_estimation import DEX


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path", "-i", help="Input video path or webcam index (default=0)", type=str, default='/Users/wanghui/Desktop/6-work/D24H-fulltime/work/14-age_woman/UTKFace_3part_onlyAsianWomen'
    )
    parser.add_argument("--output", "-o", help="Output file path", default=None)

    parser.add_argument(
        "--benchmark",
        "-b",
        help="Enable benchmark mode for CUDNN",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-display",
        help="No display if processing a video file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--threshold",
        "-t",
        help="Detection threshold (default=0.8)",
        type=float,
        default=0.8,
    )
    # choices=['rtnet50', 'rtnet101', 'resnet50'])
    parser.add_argument("--encoder", "-e", help="Method to use", default="resnet50")

    parser.add_argument(
        "--loss",
        help="Method to use, can be either rtnet50 or rtnet101 (default=rtnet50)",
        default="dex",
    )
    parser.add_argument(
        "-an", "--age-classes", help="Age classes (default=101)", type=int, default=101
    )
    parser.add_argument("--max-num-faces", help="Max number of faces", default=50)
    parser.add_argument(
        "--weights",
        "-w",
        help="Weights to load, can be either resnet50 or mobilenet0.25 when using RetinaFace",
        default=None,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device to be used by the model (default=cuda:0)",
        default="cuda:0",
    )
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark
    # args.method = args.method.lower().strip()

    face_detector = RetinaFacePredictor(
        threshold=args.threshold,
        device=args.device,
        model=(RetinaFacePredictor.get_model("mobilenet0.25")),
    )
    age_estimator = DEX(
        device=args.device,
        ckpt=args.weights,
        encoder=args.encoder,
        loss=args.loss,
        age_classes=args.age_classes,
    )

    print("Face detector created using RetinaFace.")

    prediected_age_list = []
    ground_truth_list = []
    for image_name in os.listdir(args.input_path):
        name_split = image_name.split('_')
        if name_split[1] == '1' and name_split[2] == '2':
            image = cv2.imread(os.path.join(args.input_path, image_name))

            faces = face_detector(image, rgb=False)
            if len(faces) == 0:
                continue
            # Parse faces
            # print(image.shape, faces.shape) # (115, 204, 3) (1, 15)
            age = age_estimator.predict_img(image, faces, rgb=False)
            # print(age.shape) # torch.Size([1])
            # print(int(age.item())) # 20
            for a in age:
                prediected_age_list.append(int(a.item()))
                ground_truth_list.append(int(name_split[0]))

    print(ground_truth_list[:10], prediected_age_list[:10])
    test_CA(ground_truth_list, prediected_age_list)


if __name__ == "__main__":
    main()
