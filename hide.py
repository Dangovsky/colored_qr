from PIL import Image
import qrcode


def generate_qr(mess, color, filename = None):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=10,
        border=4,
    )
    qr.add_data(mess)
    qr.make(fit=True)

    img = qr.make_image(fill_color=color, back_color='white')

    if filename not is None
        img.save(filename + '.jpeg', 'jpeg')
    return img


def four_color_hide(mess, file_name = None):
    firstpart, secondpart = mess[:round(
        len(mess)/2)], mess[round(len(mess)/2):]

    qr_c = generate_qr(firstpart, 'cyan')
    qr_m = generate_qr(secondpart, 'magenta')
    summ = Image.merge('RGB', (qr_c.getchannel(
        0), qr_m.getchannel(1), qr_c.getchannel(2)))
    if filename not is None
        summ.save(file_name + 'two.jpeg', 'jpeg')
    return summ


def eight_color_hide(mess, file_name = None):
    firstpart, secondpart, thirdpart = mess[:round(len(
        mess)/3)], mess[round(len(mess)/3):round(len(mess)/3*2)], mess[round(len(mess)/3*2):]

    qr_r = generate_qr(thirdpart, 'cyan')
    qr_g = generate_qr(secondpart, 'magenta')
    qr_b = generate_qr(firstpart, 'yellow')
    summ = Image.merge('RGB', (qr_r.getchannel(
        0), qr_g.getchannel(1), qr_b.getchannel(2)))
    if filename not is None
        summ.save(filename + '.jpeg', 'jpeg')
    return summ
