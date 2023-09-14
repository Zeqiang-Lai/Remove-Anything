from setuptools import setup, find_packages

setup(
    name='remove_anything',
    packages=find_packages(),
    version='0.0.2',
    package_data={
        'remove_anything': [
            'mat/torch_utils/ops/bias_act.cpp',
            'mat/torch_utils/ops/upfirdn2d.cpp',
            'mat/torch_utils/ops/bias_act.cu',
            'mat/torch_utils/ops/upfirdn2d.cu',
            'mat/torch_utils/ops/bias_act.h',
            'mat/torch_utils/ops/upfirdn2d.h',
            'fcf/torch_utils/ops/bias_act.cpp',
            'fcf/torch_utils/ops/upfirdn2d.cpp',
            'fcf/torch_utils/ops/bias_act.cu',
            'fcf/torch_utils/ops/upfirdn2d.cu',
            'fcf/torch_utils/ops/bias_act.h',
            'fcf/torch_utils/ops/upfirdn2d.h',
            'ldm/config.yaml',
        ],
    },
    include_package_data=True,
    requires=['dola']
)
