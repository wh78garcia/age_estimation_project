running command:
      1）install the required library:
            pip install -r requirements.txt
      2）put the pretrained model(resnet50-dldl-101.pth.tar) to ibug/age_estimation/weights folder
      3）
      when there is cuda device:
            python test.py --input_path /the-imgs-path --device cuda:0
      when there is no cuda device:
            python test.py --input_path /the-imgs-path