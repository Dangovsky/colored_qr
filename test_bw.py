

def bw_make_test(
    repeats=10,
    noise_types=['gauss', 'salt and pepper', 'poisson'],
    gauss_sigma_step=5,
    s_p_amount_step=0.002,
    poisson_peak_step=5,
    ALPHABET=np.array(list(string.ascii_lowercase + ' '))
):

    columns = ['time',
               'noise_type', 'noise_param',
               'decoded']
    data = [[None,
             None, None,
             None]]

    mess = ''.join(s for s in np.random.choice(ALPHABET, size=3 + 100))

    generate_qr(mess, 'black', filename='bw_test')
    reader = zxing.BarCodeReader()

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

            if 'gauss' == noise_type:
                sigma = 0
            if 'salt and pepper' == noise_type:
                amount = 0
            if 'poisson' == noise_type:
                peak = peak_max

            step = 0
            qr_is_finded = 1
            while 1 == qr_is_finded:
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

                    img = cv2.imread('bw_test.png')

                    if 'gauss' == noise_type:
                        img = gauss_noise(img, mean, sigma)
                    if 'salt and pepper' == noise_type:
                        img = s_p_noise(img, s_vs_p, amount)
                    if 'poisson' == noise_type:
                        img = poisson_noise(img, peak)

                    cv2.imwrite('bw_test_tmp.png', img)

                    start = time()
                    zxing_code = reader.decode('bw_test_tmp.png', True)
                    end = time()
                    time_past = end-start
                    # data_ret, time_past = hard_decode_colors(
                    #    'test_tmp.png', colors, block_size)

                    qr_is_finded = 0
                    if zxing_code is not None:
                        if mess in zxing_code.raw:
                            qr_is_finded = 1

                    data = np.append(data, [time_past,
                                            noise_type, noise_param,
                                            qr_is_finded], axis=0)

    df = pd.DataFrame(data, columns=columns)
    df = df.dropna()
    df = df.astype({'time': float,
                    'noise_type': str, 'noise_param': float,
                    'decoded': int})
    return df
