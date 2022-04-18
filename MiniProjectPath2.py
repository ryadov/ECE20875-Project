import pandas
'''
 The following is the starting code for path2 for data reading to make your first step easier.
 'dataset_2' is the clean data for path2.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_2 = df
# print(dataset_2[0:35].to_string()) #This line will print out the first 35 rows of your data
# print(dataset_2[0:35]['numPauses'])#This line will print out the first 35 rows of number of Pauses field

# Function: get_by_VidID()
# Input   : 
#           behavior-performance.txt as a Pandas Dataframe
# Output  : 
#           vid_list (list) : A list of all VidIDs in the dataframe
#           vid_data (dict) : A dictionary, where the key is a VidID and Value is a Dataframe with ONLY the values for the selected VidID

def get_by_VidID(df):

  vid_list = list(set(list(df['VidID'])))
  vid_data = dict()
  for element in vid_list:
    vid_data[element] = df[df['VidID']==element]
  return vid_list,vid_data

"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""
