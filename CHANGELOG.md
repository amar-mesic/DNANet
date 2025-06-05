# Log of changes to DNANet
Done by Amar for NFI Internship

### First Commit
* Introduce support for different [lane standards](DNAnet/data/kit_compatibility/lane_standards.py), namely WEN ILS and GENESCAN 600 LIZ.
* Make [peak validation](DNAnet/data/utils.py) more flexible.
* Remove necessity for an [annotations](DNAnet/data/data_models/hid_image.py) file. If a dataset only has the ground truth, that should be enough.
* Add a simpler (but not too simple) [CustomHIDDataset](DNAnet/data/data_models/custom_hid_dataset.py), which is the first step towards building a modular codebase using the strategy software pattern.
* Add strategies for [file categorization](DNAnet/data/parsing/file_categorization_strategy.py) and [validation of mixture samples](DNAnet/data/validation/sample_validation_strategy.py)