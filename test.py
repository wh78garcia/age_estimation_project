import os
import cv2
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.age_estimation import DEX

def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path", "-i", help="Input images path", type=str, default='')
    parser.add_argument(
        "--benchmark",
        "-b",
        help="Enable benchmark mode for CUDNN",
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
        default="dldl",
    )
    parser.add_argument(
        "-an", "--age-classes", help="Age classes (default=101)", type=int, default=101
    )
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
        default="cpu",
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
    print(args) # Namespace(age_classes=101, benchmark=False, device='cpu', encoder='resnet50', input_path='/Users/UTKFace_3part_onlyAsianWomen_test', loss='dldl', threshold=0.8, weights=None)

    for image_name in os.listdir(args.input_path):
        image = cv2.imread(os.path.join(args.input_path, image_name))
        faces = face_detector(image, rgb=False)
        if len(faces) == 0:
            print('there is no face on the image {}'.format(image_name))
            continue

        # Parse faces
        age = age_estimator.predict_img(image, faces, rgb=False)

        # only return the smallest predicted age
        print(image_name, min([a.item() for a in age]))

if __name__ == "__main__":
    main()
