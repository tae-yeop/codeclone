import csv
import json
import os

import datasets

# 다중 configuration을 위해
# Provide the links to download the images and labels in data_url and metadata_urls
class Food101Config(datasets.BuilderConfig):
    def __init__(self, data_url, metadata_urls, **kwargs):
        """
        BuilderConfig for Food
        
        """
        super(Food101Config, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.data_url = data_url
        self.metadata_urls = metadata_urls

        

# GeneratorBasedBuilder는 dict generator로부터 datasets을 만듬
class Food101(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        Food101Config(
            name='breakfast',
            description='Food types commonly eaten during breakfast.',
            data_url= "",# 아마존 같은 URL있어야 하는듯,
            metadata_urls={
                "train" : "https://link-to-breakfast-foods-train.txt", 
                "validation" : "https://link-to-breakfast-foods-validation.txt"
            }
        )
    ]
    def _info(self):
        # stores information about your dataset like its description, license, and features.
    def _split_generators(self, dl_manager):
        # downloads the dataset and defines its splits.
    def _generate_examples(self, images, metadata_path):
        # generates the images and labels for each split.
        