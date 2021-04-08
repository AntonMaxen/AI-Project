import fastai
from fastai.metrics import error_rate
from fastai.vision import ImageDataBunch, get_transforms, imagenet_stats, models, open_image, cnn_learner, data
import pandas as pd


def main():
    art_subset_2_df = pd.read_pickle('art_subset_df_2.pkl')

    art_subset_2_df = pd.DataFrame(art_subset_2_df)

    t = ImageDataBunch.from_df(
        df=art_subset_2_df, path='./data/train_2', label_col='style', fn_col='new_filename',
        ds_tfms=get_transforms(), size=90, bs=30)
    try:
        t = t.data.normalize(stats=imagenet_stats)
    except:
        print()

    print()
    learner = cnn_learner(t, models.resnet50, metrics=[error_rate])
    print()
    learner.fit_one_cycle(6)

    learner.lr_find()
    learner.recorder.plot()

    learner.unfreeze()
    learner.fit_one_cycle(7, max_lr=slice(1e-6, 3e-4))

    img = open_image('data/test_1.jpg')
    pred = learner.predict(img);
    pred
    print()


if __name__ == '__main__':
    main()
