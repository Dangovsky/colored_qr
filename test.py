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
    repeats=10,
    noise_types=['gauss', 'salt and pepper', 'poisson'],
    gauss_sigma_step=5,
    s_p_amount_step=0.002,
    poisson_peak_step=5,
    block_size_min=1,
    block_size_max=5,
    colors=8,
    ALPHABET=np.array(list(string.ascii_lowercase + ' '))
):

    columns = ['img_name', 'decoder_type', 'time',
               'block_size', 'noise_type', 'noise param',
               'yellow_qr', 'magenta_qr', 'cian_qr']
    data = [[None, None, None,
             None, None, None,
             None, None, None]]

    image_names, mess = generate_images(n_images, ALPHABET)

    for image in range(n_images):
        print(' >>> image: ', image_names[image])

        mess_len_third = len(mess[image])/3
        splitted_mess = [mess[image][:round(mess_len_third)],
                         mess[image][round(mess_len_third):round(
                             mess_len_third*2)],
                         mess[image][round(mess_len_third*2):]]

        for noise_type in noise_types:
            print(' >> {0}'.format(noise_type))

            if 'gauss' == noise_type:
                mean = 0
                sigma_step = gauss_sigma_step
                print('\tmean = {0}, sigma_step = {1}'.format(
                    mean, sigma_step))

            if 'salt and pepper' == noise_type:
                s_vs_p = 0.5
                amount_step = s_p_amount_step
                print('\ts_vs_p = {0}, amount_step = {1}'.format(
                    s_vs_p, amount_step))

            if 'poisson' == noise_type:
                peak_max = 30
                peak_step = - poisson_peak_step
                print(
                    '\tpeak_max = {0} peak_step = {1}'.format(peak_max, peak_step))

            for block_size_axis in range(block_size_min, block_size_max + 1):
                block_size = (block_size_axis, block_size_axis)
                print(' > block_size = {0}'.format(block_size))

                if 'gauss' == noise_type:
                    sigma = 0
                if 'salt and pepper' == noise_type:
                    amount = 0
                if 'poisson' == noise_type:
                    peak = peak_max

                step = 0
                qr_is_finded = [1, 1, 1]
                while sum(qr_is_finded) > 1:
                    log = '\tstep: {0}'.format(step)
                    step += 1

                    noise_param = 0.0
                    if 'gauss' == noise_type:
                        sigma += sigma_step
                        log += ', sigma = {0}'.format(sigma)
                        noise_param = sigma
                    if 'salt and pepper' == noise_type:
                        amount += amount_step
                        log += ', amount = {0}'.format(amount)
                        noise_param = amount
                    if 'poisson' == noise_type:
                        peak += peak_step
                        log += '. peak = {0}'.format(peak)
                        noise_param = peak

                    print(log)

                    for repeat in range(repeats):
                        print('\t\trepeat: {1}'.format(step, repeat))

                        img = cv2.imread(image_names[image] + '.png')

                        if 'gauss' == noise_type:
                            img = gauss_noise(img, mean, sigma)
                        if 'salt and pepper' == noise_type:
                            img = s_p_noise(img, s_vs_p, amount)
                        if 'poisson' == noise_type:
                            img = poisson_noise(img, peak)

                        cv2.imwrite('test_tmp.png', img)

                        data_ret, time_past = hard_decode_colors(
                            'test_tmp.png', colors, block_size)

                        qr_is_finded = [0, 0, 0]
                        if len(data_ret) != 0:
                            for i in range(len(splitted_mess)):
                                if splitted_mess[i] in data_ret:
                                    qr_is_finded[i] = 1

                        data = np.append(data, [[image_names[image], 'hard', time_past,
                                                 block_size_axis, noise_type, noise_param,
                                                 qr_is_finded[0], qr_is_finded[1], qr_is_finded[2]]], axis=0)

    df = pd.DataFrame(data, columns=columns)
    df = df.dropna()
    df = df.astype({'img_name': str, 'decoder_type': str, 'time': float,
                    'block_size': int, 'noise_type': str, 'noise param': float,
                    'yellow_qr': int, 'magenta_qr': int, 'cian_qr': int})

    df['decoded'] = (df['yellow_qr'] + df['magenta_qr'] + df['cian_qr']) == 3

    return df
