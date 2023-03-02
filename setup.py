from setuptools import find_packages, setup

config = {
    "version": "0.0.1",
    "name": "esp-prediction",
    "description": "ESP prediction",
    "packages": find_packages(
        include=[
            "ESP_DNN",
            "ESP_DNN.*",
        ]
    ),
}

setup(**config)
