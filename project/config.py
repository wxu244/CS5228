FLAT_MODEL_MAPPING: dict = {
    '2 room': 'Small',
    'new generation': 'Small',

    'simplified': 'Medium',
    'model a2': 'Medium',
    'standard': 'Medium',

    'model a': 'Standard',
    'improved': 'Standard',
    'premium apartment': 'Standard',

    'apartment': 'Large',
    '3gen': 'Large',
    'improved maisonette': 'Large',
    'adjoined flat': 'Large',
    'maisonette': 'Large',
    'model a maisonette': 'Large',
    'dbss': 'Large',
    'premium maisonette': 'Large',

    'terrace': 'Extra Large',
    'multi generation': 'Extra Large',

    'premium apartment loft': 'Premium',

    'type s1': 'Suite',
    'type s2': 'Suite',
}

# 数值特征
NUMERICAL_FEATURES = ['YEAR', 'FLOOR', 'AGE', 'FLOOR_AREA_SQM', 'RESALE_PRICE']

# 类别特征
CATEGORICAL_FEATURES = ['TOWN', 'FLAT_TYPE_ORIGINAL', 'FLAT_MODEL']

AUX_FILE_NAMES = [
    'sg-gov-hawkers', 'sg-hdb-block-details', 'sg-mrt-stations',
    'sg-primary-schools', 'sg-secondary-schools', 'sg-shopping-malls'
]

# 绘图配置
TOP_N_CATEGORIES = 15