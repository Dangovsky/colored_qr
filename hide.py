from PIL import Image
import qrcode


def generate_qr(filename, mess, color):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=10,
        border=4,
    )
    qr.add_data(mess)
    qr.make(fit=True)

    img = qr.make_image(fill_color=color, back_color='white')
    img.save(filename + '.jpeg', 'jpeg')
    return img


def two_color_hide(mess):
    firstpart, secondpart = mess[:round(len(mess)/2)], mess[round(len(mess)/2):]
        
    qr_c = generate_qr('cyan_2', firstpart, 'cyan')
    qr_m = generate_qr('magenta_2', secondpart, 'magenta')
    summ = Image.merge('RGB', (qr_c.getchannel(0), qr_m.getchannel(1), qr_c.getchannel(2)))
    summ.save('two.jpeg', 'jpeg')
    return summ


def three_color_hide(mess):
    firstpart, secondpart, thirdpart = mess[:round(len(mess)/3)], mess[round(len(mess)/3):round(len(mess)/3*2)], mess[round(len(mess)/3*2):]
        
    qr_r = generate_qr('red_3', firstpart, 'cyan')
    qr_g = generate_qr('green_3', secondpart, 'magenta')
    qr_b = generate_qr('blue_3', thirdpart, 'yellow')
    summ = Image.merge('RGB', (qr_r.getchannel(0), qr_g.getchannel(1), qr_b.getchannel(2)))
    summ.save('three.jpeg', 'jpeg')
    return summ
