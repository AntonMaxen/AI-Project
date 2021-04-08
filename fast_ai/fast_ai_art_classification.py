import fastai
from fastai.metrics import error_rate
from fastai.vision import ImageDataBunch, get_transforms, imagenet_stats, models, open_image, cnn_learner
import pandas as pd


art_df = pd.read_pickle('art_subset_train_2.pkl')
print()



data = ImageDataBunch.from_df(
    df=art_df, path='./data/train_2', label_col='style', fn_col='new_filename',
    ds_tfms=get_transforms(), size=299, bs=48).data.normalize(stats=imagenet_stats)


learner = cnn_learner(data, models.resnet50, metrics=[error_rate])

learner.fit_one_cycle(6)

learner.lr_find()
learner.recorder.plot()

learner.unfreeze()
learner.fit_one_cycle(7, max_lr=slice(1e-6, 3e-4))

# img = open_image('data/the_scream.jpg')
# pred = learner.predict(img); pred
