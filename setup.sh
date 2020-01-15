!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d timetraveller98/bengaliai-128
!unzip bengaliai-128.zip
!rm bengaliai-128.zip
!mkdir data
!mkdir data/train_128_feather
!mv *feather data/train_128_feather

!kaggle datasets download -d timetraveller98/traincsv
!unzip traincsv.zip
!rm traincsv.zip
!mv train.csv data/

!git clone https://<Username>:<password>@github.com/TimeTraveller-San/BengaliAI
!mv BengaliAI/* .

!ls
