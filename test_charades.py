from vocab import Vocabulary
import evaluation_charades as evaluation
DATA_PATH = '/data/usr/datasets/Text_Video_Moment/'
RUN_PATH = '/home/usr/python/weak_supervised_video_moment/runs/'

evaluation.evalrank(RUN_PATH+"test_charades/model_best.pth.tar", data_path=DATA_PATH, split="test")
