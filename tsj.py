import time
from io import BytesIO
from PIL import Image
import simplejson
import requests as req
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import tensorflow as tf
from captcha.image import ImageCaptcha
from captcha.image import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageOps

import string
characters = string.digits + string.ascii_letters
width, height, n_len, n_class = 128, 64, 4, len(characters)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# define data generator


class CaptchaSequence(Sequence):
    def __init__(self, pil_img, characters=string.digits+string.ascii_letters, batch_size=1, steps=1, n_len=4, width=128, height=64):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        #self.generator = ImageCaptcha(width=width, height=height)
        self.img = pil_img
        size = (width, height)
        self.img = ImageOps.fit(self.img, size, Image.ANTIALIAS)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height,
                      self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8)
             for i in range(self.n_len)]
        for i in range(self.batch_size):
            random_str = ''.join([random.choice(self.characters)
                                  for j in range(self.n_len)])
            X[i] = np.array(self.img) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y


# cnn

input_tensor = Input((height, width, 3))
x = input_tensor
for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
    for j in range(n_cnn):
        x = Conv2D(32*2**min(i, 3), kernel_size=3, padding='same',
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = [Dense(n_class, activation='softmax', name='c%d' % (i+1))(x)
     for i in range(n_len)]
model = Model(inputs=input_tensor, outputs=x)


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


try:
    model.load_weights('/cnn.h5')
except OSError:
    model.load_weights('cnn.h5')


ts_cap_url = "https://tushare.pro/captcha?action=register"
HOST = "https://tushare.pro"
headers_raw = [
    {
        "name": "Sec-Fetch-Mode",
        "value": "cors"
    },
    {
        "name": "Sec-Fetch-Site",
        "value": "same-origin"
    },
    {
        "name": "Origin",
        "value": "https://tushare.pro"
    },
    {
        "name": "Accept-Encoding",
        "value": "gzip, deflate, br"
    },
    {
        "name": "Host",
        "value": "tushare.pro"
    },
    {
        "name": "Accept-Language",
        "value": "en-US,en;q=0.9,zh-CN;q=0.8,zh-TW;q=0.7,zh;q=0.6"
    },
    {
        "name": "User-Agent",
        "value": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
    },
    {
        "name": "Content-Type",
        "value": "application/x-www-form-urlencoded; charset=UTF-8"
    },
    {
        "name": "Accept",
        "value": "application/json, text/javascript, */*; q=0.01"
    },
    {
        "name": "Referer",
        "value": "https://tushare.pro/register?reg=305500"
    },
    {
        "name": "X-Requested-With",
        "value": "XMLHttpRequest"
    },
    {
        "name": "Connection",
        "value": "keep-alive"
    },
    {
        "name": "DNT",
        "value": "1"
    }
]

headers = {dt['name']: dt['value'] for dt in headers_raw}
ts_s = req.session()
ts_s.headers = headers


def fet_captcha(fname):
    print('fetching captcha')
    def _req_ts(fname=fname):
        res = ts_s.get(ts_cap_url).content
        res = simplejson.loads(res)

        HOST = "https://tushare.pro"
        new_cap_url = HOST + res['data']

        img_r = ts_s.get(new_cap_url)

        img = Image.open(BytesIO(img_r.content))
        img.save(fname + '.png')
        data = CaptchaSequence(img)
        X, y = data[0]
        y_pred = model.predict(X)
        print('predictation:{0}'.format(decode(y_pred)))

        VERIFY_URL = "https://tushare.pro/captcha"
        cap_verify = res['data'].replace('captcha?', 'verify?')
        cap_verify = VERIFY_URL + cap_verify + '&captcha=' + decode(y_pred)
        # print(cap_verify)
        res_verify = ts_s.get(cap_verify).content
        res_verify = simplejson.loads(res_verify)

        return {'res_verify': res_verify, 'pred': decode(y_pred)}

    while True:
        res = _req_ts('1')
        print(repr(res))
        if res['res_verify']['code'] == 0:
            return res['pred']

def reg(captcha, refer_id='305500'):
    print('registering new user')
    mail_s = req.session()
    mail = simplejson.loads(mail_s.post(
        "https://api.internal.temp-mail.io/api/v2/email/new").content)['email']
    print(mail)

    fetch_vcode = ts_s.get("https://tushare.pro/regcode?account=" + mail)
    fetch_vcode.status_code

    i=0
    while True and i<=8:
        message = mail_s.get(
            "https://api.internal.temp-mail.io/api/v2/email/" + mail + "/messages")
        msg = simplejson.loads(message.content)

        try:
            import re
            p = re.compile(r'[0-9]{6}')
            import time
            v_code = p.findall(msg[0]['body_text'])[0]
            if v_code:
                print(v_code)
                break
        except:
            time.sleep(1)
            print('retry...')
            i += 1
            continue

    reg_url = "https://tushare.pro/register"
    params = {
        'reg': refer_id
    }
    data = {
        "account": mail,
        "password": 'JUhDcmwmXpyG28Q',
        "captcha": captcha,
        "verify_code": v_code
    }

    ts_s.get("https://tushare.pro/register", params={'reg': refer_id})
    reg_r = ts_s.post(reg_url, params=params, data=data)
    if reg_r.status_code == 200:
        print('reg OK')
    return reg_r.status_code

i = 0
while True:
    print('*'*60)
    captcha = fet_captcha(fname='1')
    try:
        reg_res = reg(captcha)

        if reg_res == 200 and i<=10:
            i += 1
            print(i)
    except:
        pass

