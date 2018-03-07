import cv2
import mxnet.ndarray as nd
import mxnet as mx

def data_preprocessing(content_image=None, style_image=None, image_size=(), ctx=None):

    # 1. content_image
    # (1) using opencv
    ci = cv2.imread(content_image, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(ci)
    ci = cv2.merge([r, g, b])

    #for faster learning and to prepare for the problem that the image is too small or too large
    ci = cv2.resize(ci, dsize=(image_size[1],image_size[0]), interpolation=cv2.INTER_AREA)

    # (2) normalization
    ci = nd.array(ci, ctx=ctx)
    ci = nd.divide(ci, 255)
    ci = mx.image.color_normalize(ci,mean=nd.array([0.485, 0.456, 0.406],ctx=ctx), std=nd.array([0.229, 0.224, 0.225],ctx=ctx))
    ci = nd.transpose(ci, axes=(2, 0, 1))
    ci = ci.reshape((-1,) + ci.shape)

    # 2. style_image
    # (1) using opencv
    si = cv2.imread(style_image, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(si)
    si = cv2.merge([r, g, b])

    #for faster learning and to prepare for the problem that the image is too small or too large
    si = cv2.resize(si, dsize=(image_size[1],image_size[0]), interpolation=cv2.INTER_AREA)
    # (2) normalization
    si = nd.array(si, ctx=ctx)
    si = nd.divide(si, 255)
    si = mx.image.color_normalize(si,mean=nd.array([0.485, 0.456, 0.406],ctx=ctx), std=nd.array([0.229, 0.224, 0.225],ctx=ctx))
    si = nd.transpose(si, axes=(2, 0, 1))
    si = si.reshape((-1,) + si.shape)

    # 3.noise image
    noise=nd.random_uniform(low=0, high=255, shape=image_size+(3,), ctx=ctx)
    noise = nd.divide(noise, 255)
    noise = mx.image.color_normalize(noise,mean=nd.array([0.485, 0.456, 0.406],ctx=ctx), std=nd.array([0.229, 0.224, 0.225],ctx=ctx))
    noise = nd.transpose(noise, axes=(2, 0, 1))
    noise = noise.reshape((-1,) + noise.shape)
    return ci, si, noise