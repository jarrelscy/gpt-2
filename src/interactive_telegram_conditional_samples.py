#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf
import time
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram_token import token
import model, sample, encoder
import traceback, logging
        
if __name__ == '__main__':

    updater = Updater(token=token)    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='telegram.log',
                    filemode='a')    
    logger = logging.getLogger(__name__)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    model_name='117M'
    seed=None
    nsamples=1
    batch_size=None
    length=40
    temperature=1
    top_k=40
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = [sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        
        def echo(bot, update):
            raw_text = update.message.text
            context_tokens = enc.encode(raw_text)
            out = sess.run(output[0], feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
            text = ('\n'+"=" * 40+'\n').join([enc.decode(o) for o in out])
            update.message.reply_text(text)
        
        def set_length(bot, update, args):
            global output
            global length
            chat_id = update.message.chat_id
            try:                
                length = int(args[0])
                assert length <= hparams.n_ctx
                output[0] = sample.sample_sequence(
                        hparams=hparams, length=length,
                        context=context,
                        batch_size=batch_size,
                        temperature=temperature, top_k=top_k
                    )

                update.message.reply_text('Length successfully set!')
            except:
                update.message.reply_text(traceback.format_exc())
                update.message.reply_text('Usage: /length <number of characters, which must be less than %s' % hparams.n_ctx)
                
        dp = updater.dispatcher
        dp.add_handler(MessageHandler(Filters.text, echo))
        dp.add_handler(CommandHandler("length", set_length,
                                  pass_args=True))
        updater.start_polling()
        while True:
            time.sleep(60)


