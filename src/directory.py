import os

# Get directories/filepaths
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

log_dir = os.path.join(project_dir, '../logs')
data_dir = os.path.join(project_dir, 'data')
figures_dir = os.path.join(project_dir, 'figures')
cph_200C_dir = os.path.join(data_dir, 'CPH 200C')
diabetic_filepath = os.path.join(cph_200C_dir, 'diabetic_data.csv')
IDS_filepath = os.path.join(cph_200C_dir, 'IDS_mapping.csv')