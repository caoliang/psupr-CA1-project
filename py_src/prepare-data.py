import random
import pandas as pd
import warnings
import numbers
import re
warnings.filterwarnings('ignore')

# Define function to count total number of records
def count_lines_in_file(data_input_file):
    line_num = 0
    with open(data_input_file, 'r', encoding='utf-8') as infile:
        line = infile.readline()
        while line:
            line_num += 1
            # Read next line
            line = infile.readline()

    return line_num

# Define function to extract records based on number of samples
def filtered_by_sample_size(num_of_samples=3500, num_of_input_file=0, data_input_file="data.csv", data_output_file="fifa_data.csv"):

    if num_of_input_file == 0:
        num_of_input_file = count_lines_in_file(data_input_file)
        print("Total records: ", num_of_input_file)

    # Randomly choose records index
    sample_index_list = random.sample(range(2, num_of_input_file), num_of_samples)

    # Read and save data by sample index
    line_num = 0
    out_line_num = 0
    with open(data_output_file, mode='w+', encoding='utf-8') as outfile:
        with open(data_input_file, 'r', encoding='utf-8') as infile:
            line = infile.readline()
            while line:
                line_num += 1
                # Show progress
                if line_num % 10000 == 0:
                    print("Processing line number: ", line_num)

                # Write title line (first line) or selected sample lines
                if line_num == 1 or line_num in sample_index_list:
                    out_line_num += 1
                    outfile.write(line)

                # Read next line
                line = infile.readline()

    # Completed
    print("Total input lines: ", line_num, ", total output lines: ", out_line_num)

def is_nan_or_empty_value(value):
    if value != value:
        return True
    if isinstance(value, str):
        if value == '' or value.strip() == '':
            return True
    return False

# Step 1. Randomly Choosing 4000 records for processing and save to CSV file
sample_size = 4000
filtered_by_sample_size(num_of_samples=sample_size, data_output_file='./../fifa19_data.csv')

# Step 2. Pre-processing data
# get_ipython().run_line_magic('matplotlib', 'inline')

# Read original data
sample_data = pd.read_csv('fifa19_data.csv', encoding='utf-8')
print('sample_data: ', sample_data.shape)
#print(sample_data.head(n=1))

# 2.1 Remove invalid data

# 2.1.1 Since this is the classification result so we will only consider valid result 'International Reputation'
drop_cond = sample_data['International Reputation'].apply(is_nan_or_empty_value)
sample_data.drop(sample_data[drop_cond].index, inplace=True)
print('sample_data: ', sample_data.shape)

# 2.2.2 Removing data without root postition scores
root_position_features = [ 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST' ]
for pos_name in root_position_features:
    drop_cond = sample_data[pos_name].apply(is_nan_or_empty_value)
    sample_data.drop(sample_data[drop_cond].index, inplace=True)
    print('clear invalid data at position score: ', pos_name, ', sample_data: ', sample_data.shape)

# 2.2 Remove features

# 2.2.1 Remove Row number column

# Rename unamed column name to 'Row_number'
row_number_col = 'Row_number'
sample_data.rename( columns={'Unnamed: 0': row_number_col}, inplace=True)
sample_data = sample_data.drop([row_number_col], axis=1, errors='ignore')
print('after removing row number column, sample_data: ', sample_data.shape)
#print(sample_data.head(n=1))

# Set index to ID column
sample_data.set_index('ID')

# 2.2.2 Remove unsed features
removed_features = [ 'Name', 'Photo', 'Flag', 'Club Logo', 'Real Face', 'Jersey Number',
                    'Joined', 'Loaned From', 'Contract Valid Until', 'Release Clause' ]
sample_data = sample_data.drop(removed_features, axis = 1, errors='ignore')
print('after removing unused features, sample data: ', sample_data.shape)
#print(sample_data.head(n=1))

# 2.2.3 Remove irrelevant features
irrelevant_features = [ 'Nationality', 'Club', 'Preferred Foot', 'Body Type', 'Position', 'Weak Foot' ]
sample_data = sample_data.drop(irrelevant_features, axis = 1, errors='ignore')
print('after removing irrelevant features, sample data: ', sample_data.shape)

# 2.2.4 Remove features represented by other features

# Since position left, center and right are condidered the same score, 
# such as LAM, CAM, and RAM are all same, so we reduce the position scores to
# the root positions only: 'ST', RS', 'RW', 'RF', 'RAM', 'RM', 'RCM', 'RWM', 'RDM', 'RB', 'RCB';
# and remove the rest of position scores
redudant_position_features = [ 'CAM', 'CB', 'CDM', 'CF', 'CM', 'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB' ]
sample_data = sample_data.drop(redudant_position_features, axis = 1, errors='ignore')
print('after removing redundant features, sample data: ', sample_data.shape)


# 2.3 Convert features to valid number
# 2.3.1 Convert money value

# Convert money string to float number
def convert_money_value(value):
    if value != value:
        return 0

    if isinstance(value, numbers.Number):
        return value

    if isinstance(value, str):
        if value == '' or value.strip() == '':
            return 0

        if isinstance(value, numbers.Number):
            return value

        if value[0] == '€':
            value = value[1:]

        if value[-1] == 'K':
            return int(float(value[:-1]) * 1000)

        if value[-1] == 'M':
            return int(float(value[:-1]) * 1000000)

        return int(value)
    else:
        return int(value)

# Convert 'Value' to market value number
sample_data['Value'] = sample_data['Value'].map(convert_money_value)
market_value_mean = int(sample_data['Value'].mean())
print('market_value_mean: ', market_value_mean)
sample_data['Value'].replace(to_replace=0, value=market_value_mean, inplace=True)
print(sample_data['Value'].head(3))
print(" ")


# Convert 'Wage' to market value number
sample_data['Wage'] = sample_data['Wage'].map(convert_money_value)
wage_value_mean = int(sample_data['Wage'].mean())
print('wage_value_mean: ', wage_value_mean)
sample_data['Wage'].replace(to_replace=0, value=wage_value_mean, inplace=True)
print(sample_data['Wage'].head(3))

# 2.3.2 Convert height value

# Convert height
def convert_height(value):
    if value != value:
        return 0

    if isinstance(value, numbers.Number):
        return value

    if not isinstance(value, str):
        #print('Invalid value: {}'.format(str(value)))
        return 0

    if value == '' or value.strip() == '':
        return 0

    height_values = value.split("'")
    h_ft = 0
    h_inch = 0
    if len(height_values) == 2:
        h_ft = int(height_values[0])
        h_inch = int(height_values[1])
    elif len(height_values) == 1:
        h_ft = int(height_values[0])

    h_total = h_ft * 12 + h_inch

    return h_total

# Convert 'Height' to Height value number
sample_data['Height'] = sample_data['Height'].map(convert_height)
height_mean = int(sample_data['Height'].mean())
print('height_mean: ', height_mean)

sample_data['Height'].replace(to_replace=0, value=height_mean, inplace=True)
print(sample_data['Height'].head(3))
print(" ")

# 2.3.3 Convert weight value

# Convert weight
def convert_weight(value):
    if value != value:
        return 0

    if isinstance(value, numbers.Number):
        return value

    if not isinstance(value, str):
        #print('Invalid value: {}'.format(str(value)))
        return 0

    if value == '' or value.strip() == '':
        return 0

    weight_value = int(value.replace('lbs', ''))

    return weight_value

# Convert 'Weight' to Weight value number
sample_data['Weight'] = sample_data['Weight'].map(convert_weight)
weight_mean = int(sample_data['Weight'].mean())
print('weight_mean: ', weight_mean)

sample_data['Weight'].replace(to_replace=0, value=weight_mean, inplace=True)
print(sample_data['Weight'].head(3))
print(" ")

# 2.3.4 Convert position value

# Convert position value to remove "+<n>" and convert to integer
extra_pos_pattern = re.compile('\\+\\d+')

def convert_position_value(position_value):
    if position_value != position_value:
        return 0

    if isinstance(position_value, numbers.Number):
        return position_value

    if not isinstance(position_value, str):
        #print('Invalid value: {}'.format(str(value)))
        return 0

    if position_value == '' or position_value.strip() == '':
        return 0

    converted_value = int(extra_pos_pattern.sub('', position_value))

    return converted_value


# Convert all value at position features
root_position_features = [ 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST' ]
for pos_name in root_position_features:
    print("Position: ", pos_name)
    sample_data[pos_name] = sample_data[pos_name].map(convert_position_value)
    pos_mean = int(sample_data[pos_name].mean())
    print('mean: ', pos_mean)
    sample_data[pos_name].replace(to_replace=0, value=pos_mean, inplace=True)
    sample_data[pos_name] = sample_data[pos_name].astype(int)
    #print(sample_data[pos_name].head(1))
    print(" ")

# 2.3.4 Convert work rate value
def convert_work_rate(score_value, work_type):
    if score_value != score_value:
        return 0

    if isinstance(score_value, numbers.Number):
        return score_value

    if not isinstance(score_value, str):
        #print('Invalid value: {}'.format(str(value)))
        return 0

    if score_value == '' or score_value.strip() == '':
        return 0

    score_levels = score_value.split('/')
    if len(score_levels) != 2:
        return 0

    level_index = 0 if work_type == 'attack' else 1

    score_text = score_levels[level_index].strip().lower()
    if score_text == 'high':
        return 3
    elif score_text == 'medium':
        return 2
    elif score_text == 'low':
        return 1
    else:
        return 0

def convert_attack_rate(score_value):
    return convert_work_rate(score_value, 'attack')

def convert_defense_rate(score_value):
    return convert_work_rate(score_value, 'defense')

# Convert attack rate
sample_data['Attack_rate'] = sample_data['Work Rate'].map(convert_attack_rate)
attack_mean = int(sample_data['Attack_rate'].mean())
sample_data['Attack_rate'].replace(to_replace=0, value=attack_mean, inplace=True)
print('attack_mean: ', attack_mean)

# Convert defense rate
sample_data['Defense_rate'] = sample_data['Work Rate'].map(convert_defense_rate)
defense_mean = int(sample_data['Defense_rate'].mean())
sample_data['Defense_rate'].replace(to_replace=0, value=defense_mean, inplace=True)
print('defense_mean: ', defense_mean)

print(sample_data[['Work Rate', 'Attack_rate', 'Defense_rate']].head(1))


# Remove Work Rate feature after conversion
sample_data = sample_data.drop('Work Rate', axis = 1, errors='ignore')
print('after removing work rate feature, sample data: ', sample_data.shape)

# 2.3.5 Convert other value

def convert_score_value(score_value):
    if score_value != score_value:
        return 0

    if isinstance(score_value, numbers.Number):
        return score_value

    if not isinstance(score_value, str):
        #print('Invalid value: {}'.format(str(value)))
        return 0

    if score_value == '' or score_value.strip() == '':
        return 0

    converted_value = int(score_value)

    return converted_value

other_scores_features = [ 'Age', 'Overall', 'Potential', 'Special' ]

for score_col in other_scores_features:
    print("score column: ", score_col)
    sample_data[score_col] = sample_data[score_col].map(convert_score_value)
    col_mean = int(sample_data[score_col].mean())
    print('mean: ', col_mean)
    sample_data[score_col] = sample_data[score_col].replace(to_replace=0, value=col_mean)
    sample_data[score_col] = sample_data[score_col].astype(int)
    #print(sample_data[score_col].head(1))
    print(" ")

# 2.3.6 Convert skill scores

skill_score_features = [ 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
                         'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
                         'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
                         'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                         'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                         'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                         'GKKicking', 'GKPositioning', 'GKReflexes' ]

for score_col in skill_score_features:
    print("score column: ", score_col)
    sample_data[score_col] = sample_data[score_col].map(convert_score_value)
    col_mean = int(sample_data[score_col].mean())
    print('mean: ', col_mean)
    sample_data[score_col] = sample_data[score_col].replace(to_replace=0, value=col_mean)
    sample_data[score_col] = sample_data[score_col].astype(int)
    #print(sample_data[score_col].head(1))
    print(" ")

# Step 3. Export pre-processed data

out_file = 'fifa19_ready_data.csv'
print('Save pre-processed sample data {} to file: {}'.format(sample_data.shape, out_file))
export_csv = sample_data.to_csv (out_file, index = None, header=True)
if export_csv is not None:
    print(export_csv)