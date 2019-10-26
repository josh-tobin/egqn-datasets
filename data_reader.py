import tensorflow as tf
from functools import partial
import logging
from os.path import join

_NUM_CHANNELS = 3

def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.uint8)

GQN_DATASET_ROOT = 'gs://gqn-dataset/'
EGQN_DATASET_ROOT = 'gs://egqn-datasets/'

class Dataset:
    def __init__(self, tf_dataset, batch_size=64, 
                 name='dataset'):
        self.name = name
        self.batch_size = batch_size
        self._dataset = tf_dataset
        self._iterator = self._dataset.make_initializable_iterator()
        self.initializer = self._iterator.initializer
        self.next_batch = self._iterator.get_next()

class GQNDataset(Dataset):
    def __init__(self, dataset_path,
                 dataset_root=GQN_DATASET_ROOT,
                 name="gqn_dataset",
                 batch_size=64,
                 sequence_size=10,
                 context_size=10,
                 image_size=64,
                 fov=50.,
                 num_camera_params=5):
        self._dataset_root = dataset_root
        self._dataset_path = dataset_path
        self._sequence_size = sequence_size
        self._context_size = context_size
        self._image_size = image_size
        self._fov = fov
        self._num_camera_params = num_camera_params

        tf_dataset = self._create_dataset(batch_size)
        super().__init__(tf_dataset, name=name)

    def to_gpu(self, gpu_id):
        gpu_dset = self._dataset.apply(tf.contrib.data.prefetch_to_device(f'/gpu:{gpu_id}', 2))
        return Dataset(gpu_dset, name=f'{self.name}_{gpu_id}',
                       batch_size=self.batch_size)

    def _create_dataset(self, batch_size):
        dataset_paths = self._get_dataset_paths(self._dataset_root, self._dataset_path)
        dataset_paths_tf = tf.data.Dataset.from_tensor_slices(dataset_paths)
        dataset_paths_tf = dataset_paths_tf.shuffle(len(dataset_paths))
        
        dataset = dataset_paths_tf.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(self._parse_example, num_parallel_calls=10)
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(8)
        return dataset

    def _get_dataset_paths(self, dataset_root, dataset_path):
        if isinstance(dataset_path, str):
            dataset_paths = join(dataset_root, dataset_path)
            if dataset_root.startswith('gs://'):
                dataset_paths = list(sorted(tf.gfile.Glob(dataset_paths)))
            else:
                dataset_paths = [datatset_paths]
        else:
            dataset_paths = [join(dataset_root, dp) for dp in dataset_path]

        return dataset_paths

    def _parse_example(self, example):
        """
        Based on the data loader code from:
        https://github.com/deepmind/gqn-datasets
        """
        feature_map = {
            'frames': tf.FixedLenFeature(
                shape=self._sequence_size, dtype=tf.string),
            'cameras': tf.FixedLenFeature(
                shape=[self._sequence_size * self._num_camera_params],
                dtype=tf.float32)
        }
        example = tf.parse_single_example(example, feature_map)
        indices = self._get_randomized_indices()
        frames = self._preprocess_frames(example, indices)
        cameras = self._preprocess_cameras(example, indices)
        result = {'context_frames': frames[:-1],
                  'context_cameras': cameras[:-1],
                  'query_camera': cameras[-1],
                  'context_fov': tf.ones([self._context_size], dtype=tf.float32) * self._fov,
                  'query_fov': tf.constant(self._fov, dtype=tf.float32),
                  'label': frames[-1]}
        return result

    def _get_randomized_indices(self):
        indices = tf.range(0, self._sequence_size)
        indices = tf.random_shuffle(indices)
        example_size = self._context_size + 1
        indices = tf.slice(indices, begin=[0], size=[example_size])
        return indices

    def _preprocess_frames(self, example, indices):
        frames = example['frames']
        frames = tf.gather(frames, indices, axis=0)
        frames = tf.map_fn(_convert_frame_data, frames, dtype=tf.uint8, back_prop=False)
        im_size = self._image_size
        img_shape = [self._context_size + 1, im_size, im_size, _NUM_CHANNELS]
        frames = tf.reshape(frames, img_shape)
        return frames

    def _preprocess_cameras(self, example, indices):
        pose = example['cameras']
        pose = tf.reshape(pose, [self._sequence_size, self._num_camera_params])
        pose = tf.gather(pose, indices, axis=0)
        # Data from rro and the original GQN datasets are stored like this
        if self._num_camera_params == 5:
            pos = pose[:, :3]
            yaw = pose[:, 3:4]
            pitch = pose[:, 4:5]
            # By design there's never any roll
            cameras = tf.concat([
                pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=-1)
            return cameras
        # Data from oab and disco is stored in the format needed downstream
        elif self._num_camera_params == 9:
            return pose
        else:
            raise ValueError(f"Unexpected number of camera params {self._num_camera_params}")

_rrc_dset = partial(GQNDataset,
                    batch_size=9, # Per gpu
                    sequence_size=10,
                    context_size=4,
                    image_size=64,
                    fov=50.,
                    dataset_root=GQN_DATASET_ROOT,
                    num_camera_params=5)
rrc_train = partial(_rrc_dset, "rooms_ring_camera/train/*",
                    name='rrc_train')
rrc_test = partial(_rrc_dset, "rooms_ring_camera/test/*",
                   name='rrc_test')
rrc_debug = partial(_rrc_dset, 
                    "rooms_ring_camera/train/0001-of-2160.tfrecord",
                    name='rrc_debug')

_rfc_dset = partial(GQNDataset,
                    batch_size=9, # Per gpu
                    sequence_size=10,
                    context_size=4,
                    image_size=128,
                    fov=50.,
                    dataset_root=GQN_DATASET_ROOT,
                    num_camera_params=5)
rfc_train = partial(_rfc_dset, 
                    "rooms_free_camera_with_object_rotations/train/*",
                    name='rfc_train')
rfc_test = partial(_rfc_dset, "rooms_free_camera_with_object_rotations/test/*",
                   name='rfc_test')
rfc_debug = partial(_rfc_dset, 
                    "rooms_free_camera_with_object_rotations/train/0001-of-2034.tfrecord",
                    name='rfc_debug')

_jaco_dset = partial(GQNDataset,
                    batch_size=9, # Per gpu
                    sequence_size=11,
                    context_size=4,
                    image_size=64,
                    fov=50.,
                    dataset_root=GQN_DATASET_ROOT,
                    num_camera_params=5)
jaco_train = partial(_jaco_dset, 
                    "jaco/train/*",
                    name='jaco_train')
jaco_test = partial(_jaco_dset, "jaco/test/*",
                   name='jaco_test')
jaco_debug = partial(_jaco_dset, 
                    "jaco/train/0001-of-3600.tfrecord",
                    name='jaco_debug')

_sm7_dset = partial(GQNDataset,
                    batch_size=9, # Per gpu
                    sequence_size=15,
                    context_size=4,
                    image_size=64,
                    fov=50.,
                    dataset_root=GQN_DATASET_ROOT,
                    num_camera_params=5)
sm7_train = partial(_sm7_dset,
                    "shepard_metzler_7_parts/train/*",
                    name="sm7_train")
sm7_test = partial(_sm7_dset,
                    "shepard_metzler_7_parts/test/*",
                    name="sm7_test")
sm7_debug = partial(_sm7_dset,
                    "shepard_metzler_7_parts/train/001-of-900.tfrecord",
                    name="sm7_debug")


_oab_dset = partial(GQNDataset,
                    batch_size=8, # Per gpu
                    sequence_size=4,
                    context_size=3,
                    image_size=128,
                    fov=21.,
                    dataset_root=EGQN_DATASET_ROOT,
                    num_camera_params=9)
oab_train = partial(_oab_dset, 
                    [f"openai-block/{i:04}-of-2500.tfrecord" for i in range(1, 2001)],
                    name='oab_train')
oab_test = partial(_oab_dset,
                   [f"openai-block/{i:04}-of-2500.tfrecord" for i in range(2001, 2501)],
                   name='oab_test')
oab_debug = partial(_oab_dset,
                   f"openai-block/0001-of-2500.tfrecord",
                   name='oab_debug')


_disco_dset = partial(GQNDataset,
                    batch_size=8, # Per gpu
                    sequence_size=4,
                    context_size=3,
                    image_size=128,
                    fov=45.,
                    dataset_root=EGQN_DATASET_ROOT,
                    num_camera_params=9)
disco_train = partial(_disco_dset,
                      [f'disco-humanoid/{i:04}-of-2500.tfrecord' for i in range(1, 2001)],
                      name='disco_train')
disco_test = partial(_disco_dset,
                      [f'disco-humanoid/{i:04}-of-2500.tfrecord' for i in range(2001, 2501)],
                      name='disco_test')
disco_debug = partial(_disco_dset,
                      f'disco-humanoid/0001-of-2500.tfrecord',
                      name='disco_debug')

_rro_dset = partial(GQNDataset,
                    batch_size=9, # Per gpu
                    sequence_size=4,
                    context_size=3,
                    image_size=128,
                    fov=50.,
                    dataset_root=EGQN_DATASET_ROOT,
                    num_camera_params=5)
rro_train = partial(_rro_dset, 
                    [f'rooms-random-objects/{i:04}-of-1943.tfrecord' for i in range(2, 1601)],
                    name='rro_train')
rro_test = partial(_rro_dset, 
                   [f'rooms-random-objects/{i:04}-of-1943.tfrecord' for i in range(1601, 1944)],
                   name='rro_test')
rro_debug = partial(_rro_dset,
                    f'rooms-random-objects/0002-of-1943.tfrecord',
                    name='rro_debug')
