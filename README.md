# Datasets used to train Generative Query Networks with Epipolar Cross Attention (E-GQNs) in the 'Geometry-Aware Neural Rendering' paper  

This repo is based on the dataset loading code from the GQN datasets found [here](https://github.com/deepmind/gqn-datasets). Note that the 
code in this repo can be used to load some of those datasets, but they must be downloaded separately. Follow the instructions in the original repo to do so. 

This code can be used to load the following datasets:

+ **rooms_ring_camera** from the original GQN paper
+ **rooms_free_camera** from the original GQN paper
+ **jaco** from the original GQN paper
+ **shepard_metzler_7_parts** from the original GQN paper
+ **openai_block**. A ShadowHand robot with a random physically sensible finger configuration is placed in the middle of the scene. A lettered cube is placed in the hand with a random orientation. 
The appearance of the scene is changed at each example by randomizing the lighting and textures of all bodies in the scene. This dataset is based on the one from the [HandManipulateBlock-v0](https://gym.openai.com/envs/HandManipulateBlock-v0/)
gym environment.
+ **disco_humanoid**. A humanoid from the [Humanoid-v2](https://gym.openai.com/envs/Humanoid-v2/) gym environment is placed in the middle of the scene. All of its joints are configured randomly. 
The appearance of the scene is changed at each example by randomizing the lighting and textures of all bodies in the scene. 
+ **rooms_random_objects**. One to four objects from the [ShapeNet](https://arxiv.org/abs/1512.03012) dataset are randomly oriented and dropped into the scene so they land with a random, but physically plausible orientation. 
The appearance of the scene is changed at each example by randomizing the lighting, texture of the walls, and textures of all of the objects.

## Usage example

To stream a dataset from Google cloud storage:

```python
dataset = rrc_debug() # Or rrc_train, oab_test, etc
with tf.Session() as sess:
    sess.run(dataset.initializer)
    while True:
        sess.run(dataset.next_batch) # gives you a new batch each time
```

If you downloaded the datasets, you need to point the dataset constructors to the
path where they are stored:

```python
dataset = rrc_debug(dataset_root="/path/to/folder_containing_datasets/")
...
```

## Download the datasets

The code is set up to stream the data directly from the GS Buckets without manually downloading it. This is a good option for getting started, or if your training will take place on Google's cloud.

If you are not training on Google cloud, this option will probably be slow. You can download the data by using `gsutil cp`. The `rooms_ring_camera`, `rooms_free_camera`, `jaco`, and `shepard_metzler_7_parts` 
datasets are located [here](https://console.cloud.google.com/storage/browser/gqn-dataset) and the `openai_block`, `disco_humanoid`, and `rooms_random_objects` datasets are located [here](https://console.cloud.google.com/storage/browser/egqn-datasets).

See the `gsutil` [documentation](https://cloud.google.com/storage/docs/gsutil_install) for more information.

## Troubleshooting

### Loading data fails or hangs

If you see an error message like:
```
The operation failed and will be automatically retried in 1.38118 seconds (attempt 1 out of 10), caused by: Unavailable: Error executing an HTTP request (HTTP response code 0, error code 6, error message 'Couldn't resolve host 'metadata'')
```

Then make sure you are logged into gcloud by running `gcloud auth application-default login`

### Other tensorflow issues

This code was tested with 1.13.1, try using that version.
