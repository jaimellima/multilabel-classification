
from preprocessingConfig import PreprocessingConfiguration

preprocConfig = PreprocessingConfiguration()

print(preprocConfig.sample)
preprocConfig.sample = 200
print(preprocConfig.sample)
print(preprocConfig.x_columns)
print(preprocConfig.y_columns)
print(preprocConfig.spacy_method)