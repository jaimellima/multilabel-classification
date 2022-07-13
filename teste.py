from config.preprocessingConfig import PreprocessingConfig
from services.preprocessing import preprocessing
from services.preprocessing.preprocessing import Preprocessing

preprocConfig = PreprocessingConfig()

preproprocessing = Preprocessing()
preproprocessing.set_dataset(preprocConfig.origin_csv_file, preprocConfig.sample)
preproprocessing.set_X(x_columns=preprocConfig.x_columns)
text = preproprocessing.get_X()
text = preproprocessing.get_preprocessed_text()
print(text)

