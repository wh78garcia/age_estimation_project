import os
import cv2
import time
import numpy as np
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.age_estimation import DEX

def test_CA(age2, age_p2):
    age2 = np.array(age2)
    age_p2 = np.array(age_p2)
    CA3=0
    CA5=0
    CA7=0
    CA9=0

    CA3_without_0_18=0
    CA5_without_0_18=0
    CA7_without_0_18=0
    CA9_without_0_18=0
    count_without_0_18 = 0

    for i in range(0,len(age2)):
      error=np.absolute(age2[i]-age_p2[i])
      if age2[i] >= 18:
        count_without_0_18 += 1
        if error<=3:
          CA3_without_0_18+=1
        if error<=5:
          CA5_without_0_18+=1
        if error<=7:
          CA7_without_0_18+=1
        if error<=9:
          CA9_without_0_18+=1

      if age2[i]>=-1:
        if error<=3:
          CA3+=1
        if error<=5:
          CA5+=1
        if error<=7:
          CA7+=1
        if error<=9:
          CA9+=1

    CA3/=len(age2[age2>=-1])
    CA5/=len(age2[age2>=-1])
    CA7/=len(age2[age2>=-1])
    CA9/=len(age2[age2>=-1])

    CA3_without_0_18/=count_without_0_18
    CA5_without_0_18/=count_without_0_18
    CA7_without_0_18/=count_without_0_18
    CA9_without_0_18/=count_without_0_18

    print('number of all: {}, number witout 0-17: {}'.format(len(age2[age2>=-1]), count_without_0_18))
    print('CA3: ',CA3,'\nCA5: ',CA5,'\nCA7: ',CA7,'\nCA9: ',CA9)
    print('CA3_without_0_18: ',CA3_without_0_18,'\nCA5_without_0_18: ',CA5_without_0_18,'\nCA7_without_0_18: ',CA7_without_0_18,'\nCA9_without_0_18: ',CA9_without_0_18)


def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path", "-i", help="Input video path or webcam index (default=0)", type=str, default='/home/d24h_prog1/Mia/age_woman/UTKFace_3part_onlyAsianWomen'
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
    print(args)
    prediected_age_list = []
    ground_truth_list = []
    for image_name in os.listdir(args.input_path):
        if 'UTKFace' in args.input_path:
            name_split = image_name.split('_')
            if name_split[1] == '1' and name_split[2] == '2':
                image = cv2.imread(os.path.join(args.input_path, image_name))
                ground_truth_age = int(name_split[0])
            else:
                continue
        else:
            image = cv2.imread(os.path.join(args.input_path, image_name))
            ground_truth_age = int(image_name.split('.')[0].split('_')[-1])
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
            ground_truth_list.append(ground_truth_age)

    print(ground_truth_list[:10], prediected_age_list[:10])
    test_CA(ground_truth_list, prediected_age_list)


if __name__ == "__main__":
    main()
