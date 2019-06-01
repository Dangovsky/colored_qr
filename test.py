import string
import numpy as np
import pandas as pd


def generate_images(n_images, ALPHABET):
    image_names = []
    mess = []

    for i in range(n_images):
        image_names.append('test_'+str(i))
        tmp = ''.join(s for s in np.random.choice(ALPHABET, size=i * 9 + 100))
        mess.append(tmp)
        three_color_hide(mess[i], image_names[i])

    return image_names, mess


def make_test(
    n_images=5,
    n_steps=5,
    o_size_divider=3,
    h_block_size=(2, 2),
    n_colors=8,
    ALPHABET=np.array(list(string.ascii_lowercase + ' '))
):

    columns = ['img_name', 'decoder_type', 'time',
               'noise_type', 'noise param',
               'yellow_qr', 'magenta_qr', 'cian_qr']
    data = [[None, None, None,
             None, None,
             None, None, None]]

    image_names, mess = generate_images(n_images, ALPHABET)

    noise_type = 'gauss'
    mean = 0
    sigma_max = 50
    sigma_step = sigma_max / (n_steps - 1)

    print(noise_type)
    print('mean = ', mean, ', sigma_max =',
          sigma_max, 'sigma_step = ', sigma_step)

    for image in range(n_images):
        print('image: ', image_names[image])

        sigma = 0
        splitted_mess = [mess[image][:round(len(mess)/3)],
                         mess[image][round(len(mess)/3):round(len(mess)/3*2)],
                         mess[image][round(len(mess)/3*2):]]

        for step in range(n_steps):
            print('\tstep: ', step, ' sigma = ', sigma)

            img = cv2.imread(image_names[image] + '.png')
            img = gauss_noise(img, mean, sigma)
            cv2.imwrite('test_tmp.png', img)

            o_data_ret, o_time_past = decode_color_qr(
                'test.png', n_colors, o_size_divider)
            h_data_ret, h_time_past = hard_decode_colors(
                'test.png', n_colors, h_block_size)

            o_qr_is_finded = [False, False, False]
            if len(o_data_ret) != 0:
                for i in range(len(splitted_mess)):
                    if splitted_mess[i] in o_data_ret:
                        o_qr_is_finded[i] = True

            print(o_data_ret)
            print(o_qr_is_finded)

            h_qr_is_finded = [False, False, False]
            if len(h_data_ret) != 0:
                for i in range(len(splitted_mess)):
                    if splitted_mess[i] in h_data_ret:
                        h_qr_is_finded[i] = True

            data = np.append(data, [[image_names[image], 'ordinary', o_time_past,
                                     noise_type, sigma,
                                     o_qr_is_finded[0], o_qr_is_finded[1], o_qr_is_finded[2]]], axis=0)

            data = np.append(data, [[image_names[image], 'hard', h_time_past,
                                     noise_type, sigma,
                                     h_qr_is_finded[0], h_qr_is_finded[1], h_qr_is_finded[2]]], axis=0)

            sigma += sigma_step

    df = pd.DataFrame(data, columns=columns)
    df = df.astype({'img_name': str, 'decoder_type': str, 'time': float,
                    'noise_type': str, 'noise param': float,
                    'yellow_qr': bool, 'magenta_qr': bool, 'cian_qr': bool})
    return df
