from PIL import Image
import qrcode
from qrcode import QRCode


def generate_qr(mess, color, version=None, filename=None):
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=10,
        border=4,
    )
    qr.add_data(mess, optimize=0)
    if version is not None:
        qr.make(fit=True)

    img = qr.make_image(fill_color=color, back_color='white')
    if filename is not None:
        img.save(filename + '.png', 'png')
    return img


def two_color_hide(mess, filename=None):
    firstpart, secondpart = mess[:round(
        len(mess)/2)], mess[round(len(mess)/2):]

    qr_c = generate_qr(firstpart, 'cyan')
    qr_m = generate_qr(secondpart, 'magenta')
    summ = Image.merge('RGB', (qr_c.getchannel(
        0), qr_m.getchannel(1), qr_c.getchannel(2)))

    if filename is not None:
        summ.save(filename + '.png', 'png')
    return summ


def three_color_hide(mess, filename=None):
    firstpart, secondpart, thirdpart = mess[:round(len(
        mess)/3)], mess[round(len(mess)/3):round(len(mess)/3*2)], mess[round(len(mess)/3*2):]

    qr_r = generate_qr(thirdpart, 'cyan')
    qr_g = generate_qr(secondpart, 'magenta')
    qr_b = generate_qr(firstpart, 'yellow')
    summ = Image.merge('RGB', (qr_r.getchannel(
        0), qr_g.getchannel(1), qr_b.getchannel(2)))

    if filename is not None:
        summ.save(filename + '.png', 'png')
    return summ
