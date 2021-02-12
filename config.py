marker_root = './Multiscale/template_marker'


def set_marker_type():
    font_name = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
                 'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']

    marker_alkareem = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin',
                       'Nun Middle', 'Nun End', 'Mim Isolated', 'Mim Begin',
                       'Mim Middle', 'Mim End 1', 'Mim End 2']

    marker_alqalam = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin',
                      'Nun Middle', 'Nun End', 'Mim Isolated', 'Mim Begin',
                      'Mim Middle', 'Mim End']

    marker_kfgqpc = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin 1',
                     'Nun Begin 2', 'Nun Middle', 'Nun End', 'Mim Isolated',
                     'Mim Begin', 'Mim Middle', 'Mim End']

    marker_lpmq = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin 1',
                   'Nun Begin 2', 'Nun Middle', 'Nun End', 'Mim Isolated',
                   'Mim Begin', 'Mim Middle', 'Mim End 1', 'Mim End 2']

    marker_pdms = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin',
                   'Nun Middle', 'Nun End', 'Mim Isolated', 'Mim Begin',
                   'Mim Middle', 'Mim End']

    marker_amiri = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin 1',
                    'Nun Begin 2', 'Nun Begin 3', 'Nun Middle', 'Nun End',
                    'Mim Isolated', 'Mim Begin', 'Mim Middle', 'Mim End 1',
                    'Mim End 2']

    marker_meQuran = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin 1',
                      'Nun Begin 2', 'Nun Middle', 'Nun End', 'Mim Isolated',
                      'Mim Begin', 'Mim Middle', 'Mim End 1', 'Mim End 2']

    marker_norehidayat = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin 1',
                          'Nun Begin 2', 'Nun End', 'Mim Isolated', 'Mim Begin',
                          'Mim Middle', 'Mim End']

    marker_norehira = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin',
                       'Nun Middle', 'Nun End', 'Mim Isolated', 'Mim Begin',
                       'Mim Middle', 'Mim End 1', 'Mim End 2']

    marker_norehuda = ['Tanwin 1', 'Tanwin 2', 'Nun Isolated', 'Nun Begin',
                       'Nun Middle', 'Nun End', 'Mim Isolated', 'Mim Begin',
                       'Mim Middle', 'Mim End']
    list_marker = [marker_alkareem, marker_alqalam, marker_kfgqpc, marker_lpmq,
                   marker_pdms, marker_amiri, marker_meQuran, marker_norehidayat,
                   marker_norehira, marker_norehuda
                   ]
    marker_type = {}
    for x in range(len(font_name)):
        marker_type[font_name[x]] = list_marker[x]

    return marker_type


def get_marker_path():
    import glob
    global marker_root

    font_folder = ['AlKareem', 'AlQalam', 'KFGQPC', 'LPMQ', 'PDMS',
                   'amiri', 'meQuran', 'norehidayat', 'norehira', 'norehuda']
    marker_path = {}
    for name in font_folder:
        temp_list = []
        temp_glob = sorted(glob.glob(marker_root + name + '/*.png'))
        for path in temp_glob:
            temp_list.append(path[35:])
        marker_path[name] = temp_list
    # for name in font_folder:
    #     marker_path[name] = sorted(glob.glob(marker_root + name + '/*.png'))

    return marker_path


class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = "B\xb2?.\xdf\x9f\xa7m\xf8\x8a%,\xf7\xc4\xfa\x91"

    # DB_NAME = "production-db"
    # DB_USERNAME = "admin"
    # DB_PASSWORD = "example"

    MARKER = set_marker_type()
    IMAGE_UPLOADS = "./Multiscale/static/img/uploads"

    SESSION_COOKIE_SECURE = True


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True
    global marker_root

    # DB_NAME = "development-db"
    # DB_USERNAME = "admin"
    # DB_PASSWORD = "example"
    # print('dev')
    MARKER_TYPE = set_marker_type()
    MARKER_FOLDER = get_marker_path()
    # MARKER_ROOT = './Multiscale/marker'
    MARKER_ROOT = marker_root
    IMAGE_UPLOADS = "./Multiscale/static/img/uploads"

    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    TESTING = True

    DB_NAME = "development-db"
    DB_USERNAME = "admin"
    DB_PASSWORD = "example"

    SESSION_COOKIE_SECURE = False
