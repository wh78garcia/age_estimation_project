running command:
      install the required library:
            pip install -r requirements.txt
      when there is cuda device:
            python test.py --input_path /the-imgs-path --device cuda:0
      when there is no cuda device:
            python test.py --input_path /the-imgs-path