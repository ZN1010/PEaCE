import re
import cv2
import argparse

import numpy as np

from itertools import accumulate


no_arg_latex_symbols = [
    '\\mu', '\\alpha', '\\partial', '\\nu', '\\pi', '\\phi', '\\pm', '\\delta', '\\lambda', '\\beta',
    '\\theta', '\\sigma', '\\gamma', '\\psi', '\\sigma', '\\tau', '\\sum', '\\omega', '\\epsilon', '\\eta',
    '\\xi', '\\Phi', '\\Gamma', '\\infty', '\\Lambda', '\\varphi', '\\Delta', '\\chi', '\\Omega', '\\kappa',
    '\\cdot', '\\Psi', '\\equiv', '\\zeta', '\\varepsilon', '\\prod', '\\Pi', '\\Theta', '\\neq', '\\circ',
    '^{\\circ}', '\\quad'
]

one_arg_latex_symbols = [
    '\\bar {} {} {}', '\\hat {} {} {}', '\\sqrt {} {} {}', '\\tilde {} {} {}', '\\sum _ {} {} {}',
    '\\vec {} {} {}', '\\prod _ {} {} {}', '\\mathring {} {} {}', '\\dot {} {} {}'
]

one_arg_latex_symbols_alphanum_only = [
    '\\mathcal{{{}}}',
]

two_arg_latex_symbols = [
    '\\frac{{{}}}{{{}}}', '\\sum_{{{}}}^{{{}}}', '\\prod_{{{}}}}^{{{}}}'
]


def find_bad_brackets(S):
    deltas   = [(c=='{')-(c=='}') for c in S] # 1 or -1 for open/close
    forward  = [*accumulate(deltas,lambda a,b:max(0,a)+b)]        # forward levels
    backward = [*accumulate(deltas[::-1],lambda a,b:min(0,a)+b)]  # backward levels
    levels   = [min(f,-b) for f,b in zip(forward,backward[::-1])] # combined levels
    return [i for i,b in enumerate(levels) if b<0]                # mismatches



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_percentiles(value_list, percentile_list):
    value_list = list(sorted(value_list))
    output = [value_list[int(len(value_list) * (percentile / 100))] for percentile in percentile_list]

    return output


def pixelate_image(img, w_offset=1, h_offset=1):
    height, width = img.shape[:2]
    w, h = (max(width - w_offset, 1), max(height - h_offset, 1))
    # Resize input to "pixelated" size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    return output


def pad_hot_img(img, n):
    img = np.concatenate([
        np.zeros((n, img.shape[1])),
        img,
        np.zeros((n, img.shape[1])),
    ], axis=0)

    img = np.concatenate([
        np.zeros((img.shape[0], n)),
        img,
        np.zeros((img.shape[0], n)),
    ], axis=1)

    return img


def cascade_add(img, n):
    sum = np.zeros((img.shape[0] - 2 * n, img.shape[1] - 2 * n))

    raw_img_top, raw_img_left = n, n
    raw_img_bot = img.shape[0] - n
    raw_img_right = img.shape[1] - n

    for i in range(n):
        sum += img[raw_img_top:raw_img_bot, raw_img_left - i:raw_img_right - i]
        sum += img[raw_img_top:raw_img_bot, raw_img_left + i:raw_img_right + i]

        sum += img[raw_img_top - i:raw_img_bot - i, raw_img_left:raw_img_right]
        sum += img[raw_img_top + i:raw_img_bot + i, raw_img_left:raw_img_right]

        sum += img[raw_img_top - i:raw_img_bot - i, raw_img_left - i:raw_img_right - i]
        sum += img[raw_img_top + i:raw_img_bot + i, raw_img_left + i:raw_img_right + i]

        sum += img[raw_img_top - i:raw_img_bot - i, raw_img_left + i:raw_img_right + i]
        sum += img[raw_img_top + i:raw_img_bot + i, raw_img_left - i:raw_img_right - i]

    return sum


def make_bold(img, n=5):
    bold_img = img + np.zeros_like(img)
    hot_img = np.zeros_like(bold_img)
    hot_img[bold_img == 0] = 1

    padded_hot_img = pad_hot_img(hot_img, n)
    cascaded_hot_img = cascade_add(padded_hot_img, n)
    cascaded_hot_img[hot_img == 1] = 0
    bold_img[cascaded_hot_img >= n] = 0

    return bold_img


def try_cast_int(x):
    try:
        x = int(x)
    except:
        pass

    return x


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {try_cast_int(k): v for k, v in x.items()}
    return x


def convert_str_to_label_format(raw_str):
    label_components = []
    # print('raw_str: {}'.format(raw_str))
    if re.match(r'\\([a-zA-Z]+)', raw_str):
        # print('*' * 40)
        # print('\tAdding {} as own token'.format(raw_str))
        # print('*' * 40)
        label_components.append(raw_str)
    else:
        for char in raw_str:
            if char == ' ':
                label_components.append('\;')
            elif char == '%':
                label_components.append('\%')
            elif char == '&':
                label_components.append('\&')
            elif char == '$':
                label_components.append('\$')
            elif char == '#':
                label_components.append('\#')
            else:
                label_components.append(char)

    label_str = ' '.join(label_components)
    return label_str
