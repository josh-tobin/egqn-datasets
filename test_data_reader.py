import tensorflow as tf
from data_reader import (rrc_train, rrc_test, rrc_debug,
                         rfc_train, rfc_test, rfc_debug,
                         jaco_train, jaco_test, jaco_debug,
                         sm7_train, sm7_test, sm7_debug,
                         oab_train, oab_test, oab_debug,
                         disco_train, disco_test, disco_debug,
                         rro_train, rro_test, rro_debug)

def render_one_batch(dset):
    with tf.Session() as sess:
        sess.run(dset.initializer)
        batch = sess.run(dset.next_batch)
    return batch


def test_rrc():
    render_one_batch(rrc_train())
    render_one_batch(rrc_test())
    render_one_batch(rrc_debug())

def test_rfc():
    render_one_batch(rfc_train())
    render_one_batch(rfc_test())
    render_one_batch(rfc_debug())

def test_jaco():
    render_one_batch(jaco_train())
    render_one_batch(jaco_test())
    render_one_batch(jaco_debug())

def test_sm7():
    render_one_batch(sm7_train())
    render_one_batch(sm7_test())
    render_one_batch(sm7_debug())

def test_oab():
    render_one_batch(oab_train())
    render_one_batch(oab_test())
    render_one_batch(oab_debug())

def test_disco():
    render_one_batch(disco_train())
    render_one_batch(disco_test())
    render_one_batch(disco_debug())

def test_rro():
    render_one_batch(rro_train())
    render_one_batch(rro_test())
    render_one_batch(rro_debug())
